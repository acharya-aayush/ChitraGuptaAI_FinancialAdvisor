"""
ChitraGupta Simple Flask App - Lightweight Version
No heavy ML dependencies, just Ollama
"""

from flask import Flask, render_template, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Import lightweight advisor
from lightweight_advisor import get_response, clear_memory, get_advisor

@app.route('/')
def index():
    """Serve chat interface"""
    return render_template('premium_chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'success': False, 'error': 'Empty message'})
        
        logger.info(f"Query: {message[:80]}...")
        
        response, metadata = get_response(message)
        
        logger.info(f"Response: {len(response)} chars, {metadata.get('response_time', 0):.1f}s")
        
        return jsonify({
            'success': True,
            'response': response,
            'metadata': metadata
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/memory/clear', methods=['POST'])
def clear():
    """Clear conversation memory"""
    try:
        clear_memory()
        return jsonify({'success': True, 'message': 'Memory cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/memory/stats')
def stats():
    """Get memory stats"""
    try:
        advisor = get_advisor()
        return jsonify({
            'success': True,
            'memory_stats': {
                'total_interactions': len(advisor.conversation_history),
                'current_industry': advisor.current_industry
            },
            'performance_stats': {
                'avg_response_time': 0,
                'memory_hits': 0
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'version': 'lightweight-v1'})

if __name__ == '__main__':
    print("=" * 60)
    print("🏛️  ChitraGupta Lightweight - Nepal Business Advisor")
    print("=" * 60)
    print("📍 Starting server at http://localhost:5000")
    print("💡 Using Ollama (llama3.2:3b) - no heavy ML deps")
    print("=" * 60)
    
    # Initialize advisor on startup
    try:
        advisor = get_advisor()
        print("✅ Advisor initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        exit(1)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
