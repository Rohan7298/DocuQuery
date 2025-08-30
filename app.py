import os
import io
import time
import uuid
import threading
import concurrent.futures
from datetime import datetime
import pdfplumber
import magic
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings
from pinecone import Pinecone, ServerlessSpec
from cohere import Client as CohereClient
import google.generativeai as genai
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "rag-app")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 10
RERANK_K = 3
MAX_FILE_SIZE = 50 * 1024 * 1024

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Mini RAG System startup')

sessions = {}
processing_status = {}

class SessionManager:
    @staticmethod
    def create_session():
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'created_at': datetime.now(),
            'documents': [],
            'namespace': f"ns_{int(time.time())}_{os.urandom(4).hex()}"
        }
        return session_id
    
    @staticmethod
    def get_session(session_id):
        return sessions.get(session_id)
    
    @staticmethod
    def add_document_to_session(session_id, filename):
        if session_id in sessions:
            sessions[session_id]['documents'].append({
                'filename': filename,
                'uploaded_at': datetime.now()
            })
            
    @staticmethod
    def cleanup_old_sessions(hours=24):
        now = datetime.now()
        expired_sessions = []
        for session_id, session_data in sessions.items():
            if (now - session_data['created_at']).total_seconds() > hours * 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del sessions[session_id]

def extract_text_from_pdf_optimized(file_stream, filename):
    text = ""
    pages_metadata = []
    
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text(
                    layout=False,
                    x_tolerance=3,
                    y_tolerance=3
                )
                
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    pages_metadata.append({
                        'page': page_num + 1,
                        'text': page_text,
                        'filename': filename
                    })
                    
                page.flush_cache()
                
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")
    
    return text, pages_metadata

def validate_file(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File size exceeds the maximum limit of {MAX_FILE_SIZE//1024//1024}MB")
    
    file_type = magic.from_buffer(file.read(2048), mime=True)
    file.seek(0)
    
    if 'pdf' not in file_type.lower():
        raise ValueError("Uploaded file is not a PDF")
    
    return True

def get_embeddings_client():
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY environment variable not set")
    return JinaEmbeddings(api_key=api_key)

def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")

    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    pc = Pinecone(api_key=api_key)

    indexes = pc.list_indexes()
    index_names = [index.name for index in indexes]

    if PINECONE_INDEX in index_names:
        index_info = pc.describe_index(PINECONE_INDEX)
        if index_info.dimension != 768:
            raise ValueError(
                f"Pinecone index '{PINECONE_INDEX}' has dimension {index_info.dimension}, "
                "but 768 is required for Jina embeddings. "
                "Please delete this index and let the application recreate it with dimension 768."
            )
    else:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=environment
            )
        )
        app.logger.info(f"Created new Pinecone index '{PINECONE_INDEX}' with dimension 768")

    return pc.Index(PINECONE_INDEX)

def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not set")
    return CohereClient(api_key=api_key)

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    return genai

def generate_embeddings_batch(texts, embeddings_client, batch_size=50):
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_embeddings = embeddings_client.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            app.logger.warning(f"Batch embedding failed, falling back to individual: {str(e)}")
            for text in batch:
                try:
                    embedding = embeddings_client.embed_query(text)
                    all_embeddings.append(embedding)
                    time.sleep(0.05)
                except Exception as inner_e:
                    app.logger.error(f"Failed to embed text: {str(inner_e)}")
                    all_embeddings.append([0] * 768)
    
    return all_embeddings

def find_source_for_chunk(chunk_text, chunks_metadata):
    if not chunks_metadata:
        return {"filename": "unknown", "page": 1}
    
    best_match = chunks_metadata[0]
    max_overlap = 0
    
    for metadata in chunks_metadata:
        source_text = metadata['text']
        overlap = sum(1 for char in chunk_text[:100] if char in source_text[:100])
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = metadata
    
    return best_match

def process_single_file(file, session_id):
    validate_file(file)
    return extract_text_from_pdf_optimized(file, file.filename)

def process_document_background(session_id, files, text):
    try:
        session_data = sessions[session_id]
        namespace = session_data['namespace']
        
        all_text = text
        all_chunks_metadata = []
        total_files = len(files)
        
        processing_status[session_id] = {
            "status": "processing",
            "progress": 0,
            "processed_files": 0,
            "total_files": total_files
        }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_file = {}
            
            for file in files:
                if file.filename == '':
                    continue
                future = executor.submit(process_single_file, file, session_id)
                future_to_file[future] = file.filename
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                filename = future_to_file[future]
                try:
                    file_text, pages_metadata = future.result()
                    all_text += "\n\n" + file_text
                    all_chunks_metadata.extend(pages_metadata)
                    SessionManager.add_document_to_session(session_id, filename)
                    
                    processing_status[session_id] = {
                        "status": "processing",
                        "progress": (i + 1) / total_files * 100,
                        "processed_files": i + 1,
                        "total_files": total_files
                    }
                    
                except Exception as e:
                    app.logger.error(f"Error processing file {filename}: {str(e)}")
                    processing_status[session_id] = {
                        "status": "error",
                        "error": f"Error processing {filename}: {str(e)}"
                    }
                    return
        
        if not all_text.strip():
            processing_status[session_id] = {
                "status": "error",
                "error": "No text content found in the documents"
            }
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.create_documents([all_text])

        embeddings_client = get_embeddings_client()
        pinecone_index = get_pinecone_client()

        texts = [chunk.page_content for chunk in chunks]
        embeddings = generate_embeddings_batch(texts, embeddings_client)

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            source_metadata = find_source_for_chunk(chunk.page_content, all_chunks_metadata)
            
            vectors.append({
                "id": f"chunk_{i}_{int(time.time())}",
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content,
                    "chunk_index": i,
                    "source": source_metadata.get('filename', 'unknown'),
                    "page": source_metadata.get('page', 1),
                    "session_id": session_id
                }
            })

        pinecone_index.upsert(vectors=vectors, namespace=namespace)

        session_data['processed_at'] = datetime.now()
        session_data['chunk_count'] = len(vectors)
        
        processing_status[session_id] = {
            "status": "complete",
            "chunk_count": len(vectors),
            "document_count": len(session_data['documents'])
        }
        
    except Exception as e:
        app.logger.error(f"Background processing error: {str(e)}")
        processing_status[session_id] = {
            "status": "error",
            "error": str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/create_session', methods=['POST'])
def create_session():
    try:
        session_id = SessionManager.create_session()
        return jsonify({"session_id": session_id})
    except Exception as e:
        app.logger.error(f"Session creation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_document():
    files = request.files.getlist('files')
    total_size = sum(len(file.read()) for file in files)
    
    for file in files:
        file.seek(0)
    
    if total_size > 20 * 1024 * 1024:
        session_id = request.form.get('session_id')
        text = request.form.get('text', '')
        
        thread = threading.Thread(
            target=process_document_background,
            args=(session_id, files, text)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "processing",
            "session_id": session_id,
            "message": "Large document processing started in background"
        })
    
    start_time = time.time()
    metrics = {"total": 0, "steps": {}, "tokens": 0}

    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({"error": "Invalid session ID"}), 400
            
        session_data = sessions[session_id]
        namespace = session_data['namespace']
        
        text = request.form.get('text', '')
        files = request.files.getlist('files')
        
        if not text and not files:
            return jsonify({"error": "No text or files provided"}), 400

        all_text = text
        all_chunks_metadata = []
        
        if files:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_file = {
                    executor.submit(process_single_file, file, session_id): file 
                    for file in files
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        file_text, pages_metadata = future.result()
                        all_text += "\n\n" + file_text
                        all_chunks_metadata.extend(pages_metadata)
                        SessionManager.add_document_to_session(session_id, file.filename)
                    except Exception as e:
                        app.logger.error(f"Error processing file {file.filename}: {str(e)}")
                        return jsonify({"error": f"Error processing {file.filename}: {str(e)}"}), 400

        if not all_text.strip():
            return jsonify({"error": "No text content found in the documents"}), 400

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.create_documents([all_text])

        embeddings_client = get_embeddings_client()
        pinecone_index = get_pinecone_client()

        texts = [chunk.page_content for chunk in chunks]
        embeddings = generate_embeddings_batch(texts, embeddings_client)

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            source_metadata = find_source_for_chunk(chunk.page_content, all_chunks_metadata)
            
            vectors.append({
                "id": f"chunk_{i}_{int(time.time())}",
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content,
                    "chunk_index": i,
                    "source": source_metadata.get('filename', 'unknown'),
                    "page": source_metadata.get('page', 1),
                    "session_id": session_id
                }
            })

        upsert_start = time.time()
        pinecone_index.upsert(vectors=vectors, namespace=namespace)
        metrics["steps"]["upsert"] = int((time.time() - upsert_start) * 1000)

        metrics["total"] = int((time.time() - start_time) * 1000)
        
        session_data['processed_at'] = datetime.now()
        session_data['chunk_count'] = len(vectors)
        
        return jsonify({
            "session_id": session_id,
            "namespace": namespace,
            "document_count": len(session_data['documents']),
            "chunk_count": len(vectors),
            "metrics": metrics
        })

    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process_status/<session_id>')
def process_status(session_id):
    if session_id in processing_status:
        return jsonify(processing_status[session_id])
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/query', methods=['POST'])
def query_document():
    start_time = time.time()
    metrics = {"total": 0, "steps": {}, "tokens": 0}

    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        session_id = data.get('session_id')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        if not session_id or session_id not in sessions:
            return jsonify({"error": "Valid session ID is required"}), 400

        session_data = sessions[session_id]
        namespace = session_data['namespace']

        embeddings_client = get_embeddings_client()
        pinecone_index = get_pinecone_client()
        cohere_client = get_cohere_client()
        gemini_client = get_gemini_client()

        embedding_start = time.time()
        query_embedding = embeddings_client.embed_query(query)
        metrics["steps"]["embedding"] = int(
            (time.time() - embedding_start) * 1000)

        retrieval_start = time.time()
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=TOP_K,
            include_metadata=True,
            namespace=namespace
        )
        metrics["steps"]["retrieval"] = int(
            (time.time() - retrieval_start) * 1000)

        rerank_start = time.time()
        texts = [match['metadata']['text'] for match in results['matches']]

        rerank_response = cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=texts,
            top_n=RERANK_K
        )
        metrics["steps"]["reranking"] = int((time.time() - rerank_start) * 1000)

        context = []
        for i, result in enumerate(rerank_response.results):
            original_index = result.index
            source_text = texts[original_index]
            source_doc = results['matches'][original_index]['metadata'].get('source', 'Unknown')
            source_page = results['matches'][original_index]['metadata'].get('page', 1)
            context.append(f"[{i+1}] Source: {source_doc} (Page {source_page})\n{source_text}")

        context_str = "\n\n".join(context)

        llm_start = time.time()

        model = gemini_client.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Always cite your sources using the provided citation format [number]. 
If the answer cannot be found in the context, say "I don't have enough information to answer this question based on the provided documents."

Context:
{context_str}

Question: {query}

Answer:"""

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1024
            }
        )

        answer_text = response.text if response.text else "I don't have enough information to answer this question based on the provided documents."

        metrics["steps"]["llm"] = int((time.time() - llm_start) * 1000)

        prompt_tokens = len(context_str) / 4 + len(query) / 4
        completion_tokens = len(answer_text) / 4
        metrics["tokens"] = int(prompt_tokens + completion_tokens)

        metrics["total"] = int((time.time() - start_time) * 1000)

        sources = []
        for i, result in enumerate(rerank_response.results):
            original_index = result.index
            sources.append({
                "id": i + 1,
                "content": texts[original_index],
                "source": results['matches'][original_index]['metadata'].get('source', 'Unknown'),
                "page": results['matches'][original_index]['metadata'].get('page', 1),
                "relevance_score": result.relevance_score
            })

        return jsonify({
            "answer": answer_text,
            "sources": sources,
            "metrics": metrics,
            "session_id": session_id
        })

    except Exception as e:
        app.logger.error(f"Query error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session_info(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
        
    return jsonify({
        "session_id": session_id,
        "created_at": sessions[session_id]['created_at'].isoformat(),
        "documents": sessions[session_id]['documents'],
        "processed_at": sessions[session_id].get('processed_at', '').isoformat() if sessions[session_id].get('processed_at') else '',
        "chunk_count": sessions[session_id].get('chunk_count', 0)
    })

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    SessionManager.cleanup_old_sessions()
    return jsonify({
        "sessions": [
            {
                "id": session_id,
                "created_at": data['created_at'].isoformat(),
                "document_count": len(data['documents'])
            } for session_id, data in sessions.items()
        ]
    })

if __name__ == '__main__':
    SessionManager.cleanup_old_sessions()
    app.run(host='0.0.0.0', port=5000, debug=True)