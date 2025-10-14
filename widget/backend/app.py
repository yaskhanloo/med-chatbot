"""
medGEMMA Backend API
Flask REST API 
"""
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import logging
from datetime import timedelta

# Import model handler (create later)
from utils.model_handler import ModelHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'medgemma-secret-key-change-in-production'  # Change for production!
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

CORS(app, supports_credentials=True)  # Enable CORS with credentials for sessions

# Initialize model handler
logger.info("Initializing model...")
model_handler = ModelHandler()
logger.info("Model ready!")


# Conversation history management
def get_conversation_history():
    """Get conversation history from session"""
    if 'conversation' not in session:
        session['conversation'] = []
    return session['conversation']


def add_to_conversation(role, content):
    """Add message to conversation history"""
    conversation = get_conversation_history()
    conversation.append({"role": role, "content": content})
    
    # Keep only last 10 messages (5 exchanges)
    if len(conversation) > 10:
        conversation = conversation[-10:]
    
    session['conversation'] = conversation
    session.modified = True


def clear_conversation():
    """Clear conversation history"""
    session['conversation'] = []
    session.modified = True


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_handler.is_loaded()
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    
    Request body:
    {
        "message": "user message here"
    }
    
    Response:
    {
        "response": "model response here",
        "status": "success"
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing message in request body',
                'status': 'error'
            }), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'error': 'Message cannot be empty',
                'status': 'error'
            }), 400
        
        logger.info(f"Received message: {user_message[:50]}...")
        
        # Get conversation history
        conversation_history = get_conversation_history()
        
        # Generate response with history
        response = model_handler.generate_response(
            user_message, 
            conversation_history=conversation_history
        )
        
        logger.info(f"Generated response: {response[:50]}...")
        
        # Add both messages to history
        add_to_conversation("user", user_message)
        add_to_conversation("assistant", response)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_conversation_endpoint():
    """Clear conversation history"""
    clear_conversation()
    return jsonify({
        'status': 'success',
        'message': 'Conversation cleared'
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
	use_reloader=False # avoid loading the model twice
    )
