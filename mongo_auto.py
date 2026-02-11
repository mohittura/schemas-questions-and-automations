import json
import os
from typing import Dict, Any, List
from datetime import datetime
from pymongo import MongoClient, ASCENDING, errors
from pathlib import Path


class MongoDBIntegration:
    """
    Store document schemas and QAs in MongoDB
    
    Collections:
    1. document_schemas - Full JSON schemas (as-is)
    2. document_qas - Optimized Q&A storage for runtime
    """
    
    def __init__(self, 
                 connection_string: str = "mongodb+srv://mohitturabit_db_user:yqAjPzqe5HuZ91Gl@cluster0.uzrpe8f.mongodb.net/",
                 database_name: str = "document_automation"):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Name of the database
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Collections
        self.schemas_collection = self.db['document_schemas']
        self.qas_collection = self.db['document_qas']
        
        # Create indexes for better performance
        self._create_indexes()
        
        print(f"✅ Connected to MongoDB: {database_name}")
    
    def _create_indexes(self):
        """Create indexes for efficient querying"""
        try:
            # Schema collection indexes
            self.schemas_collection.create_index([("document_type", ASCENDING)])
            self.schemas_collection.create_index([("document_name", ASCENDING)])
            self.schemas_collection.create_index([("_metadata.page_id", ASCENDING)], unique=True)
            
            # QA collection indexes
            self.qas_collection.create_index([("document_type", ASCENDING)])
            self.qas_collection.create_index([("schema_id", ASCENDING)])
            self.qas_collection.create_index([
                ("document_type", ASCENDING),
                ("category", ASCENDING),
                ("order", ASCENDING)
            ])
            
            print("✅ Indexes created successfully")
        except errors.OperationFailure as e:
            print(f"⚠️  Index creation warning: {e}")
    
    def store_full_schema(self, schema_data: Dict[str, Any]) -> str:
        """
        Store the complete document schema as-is
        
        Args:
            schema_data: Full JSON schema
            
        Returns:
            inserted_id: MongoDB document ID
        """
        try:
            # Add metadata
            schema_data['_storage_metadata'] = {
                'stored_at': datetime.utcnow(),
                'version': '1.0'
            }
            
            # Upsert based on page_id
            page_id = schema_data.get('_metadata', {}).get('page_id')
            
            if page_id:
                result = self.schemas_collection.update_one(
                    {'_metadata.page_id': page_id},
                    {'$set': schema_data},
                    upsert=True
                )
                
                if result.upserted_id:
                    return str(result.upserted_id)
                else:
                    # Get existing document ID
                    doc = self.schemas_collection.find_one({'_metadata.page_id': page_id})
                    return str(doc['_id']) if doc else None
            else:
                result = self.schemas_collection.insert_one(schema_data)
                return str(result.inserted_id)
                
        except Exception as e:
            print(f"❌ Error storing schema: {e}")
            return None
    
    def extract_optimized_qas(self, schema_data: Dict[str, Any], schema_id: str) -> List[Dict[str, Any]]:
        """
        Extract Q&As in optimized format for runtime use
        
        Stores only essential fields:
        - document_type
        - category
        - order
        - question_id
        - question
        - answer_type
        - required
        - options (if applicable)
        - fields (if structured_list)
        - answer (empty initially)
        - answers (empty array for structured_list)
        
        Args:
            schema_data: Full schema
            schema_id: Reference to full schema document
            
        Returns:
            List of optimized Q&A documents
        """
        optimized_qas = []
        
        document_type = schema_data.get('document_type', 'Unknown')
        document_name = schema_data.get('document_name', 'Unknown')
        
        for category in schema_data.get('question_categories', []):
            category_name = category.get('category', 'Uncategorized')
            category_order = category.get('order', 0)
            
            for idx, question in enumerate(category.get('questions', []), 1):
                # Extract only essential fields
                qa_doc = {
                    'schema_id': schema_id,
                    'document_type': document_type,
                    'document_name': document_name,
                    'category': category_name,
                    'category_order': category_order,
                    'question_order': idx,
                    'question_id': question.get('question_id', f'q_{idx}'),
                    'question': question.get('question', ''),
                    'answer_type': question.get('answer_type', 'text'),
                    'required': question.get('required', False),
                    'answer': question.get('answer', ''),
                    '_runtime_metadata': {
                        'created_at': datetime.utcnow(),
                        'last_updated': datetime.utcnow(),
                        'answered': False
                    }
                }
                
                # Add optional fields only if present
                if 'placeholder' in question:
                    qa_doc['placeholder'] = question['placeholder']
                
                if 'description' in question:
                    qa_doc['description'] = question['description']
                
                # For select/multi_select types
                if question.get('answer_type') in ['select', 'multi_select']:
                    qa_doc['options'] = question.get('options', [])
                
                # For structured_list type
                if question.get('answer_type') == 'structured_list':
                    qa_doc['fields'] = question.get('fields', [])
                    qa_doc['answers'] = question.get('answers', [])
                
                optimized_qas.append(qa_doc)
        
        return optimized_qas
    
    def store_qas(self, qas: List[Dict[str, Any]]) -> int:
        """
        Store optimized Q&As in database
        
        Args:
            qas: List of optimized Q&A documents
            
        Returns:
            Number of documents inserted
        """
        try:
            if qas:
                # Clear existing QAs for this schema_id to avoid duplicates
                if qas[0].get('schema_id'):
                    self.qas_collection.delete_many({'schema_id': qas[0]['schema_id']})
                
                result = self.qas_collection.insert_many(qas)
                return len(result.inserted_ids)
            return 0
        except Exception as e:
            print(f"❌ Error storing Q&As: {e}")
            return 0
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single JSON file and store in MongoDB
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Processing statistics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Store full schema
            schema_id = self.store_full_schema(schema_data)
            
            if not schema_id:
                return {'success': False, 'error': 'Failed to store schema'}
            
            # Extract and store optimized Q&As
            qas = self.extract_optimized_qas(schema_data, schema_id)
            qa_count = self.store_qas(qas)
            
            return {
                'success': True,
                'schema_id': schema_id,
                'qa_count': qa_count,
                'document_type': schema_data.get('document_type', 'Unknown')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_directory(self, base_dir: str = 'final_filtered_QAs'):
        """
        Process all JSON files in directory structure
        
        Args:
            base_dir: Base directory containing topic folders
        """
        print(f"\n{'='*70}")
        print("🗄️  MONGODB BATCH UPLOAD")
        print(f"{'='*70}\n")
        
        stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_qas': 0,
            'topics': {}
        }
        
        # Find all topic folders
        if not os.path.exists(base_dir):
            print(f"❌ Directory not found: {base_dir}")
            return stats
        
        topic_folders = [f for f in os.listdir(base_dir) 
                        if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.')]
        
        print(f"📁 Found {len(topic_folders)} topic folders\n")
        
        # Process each topic
        for topic_idx, topic_folder in enumerate(topic_folders, 1):
            print(f"\n{'#'*70}")
            print(f"📂 TOPIC {topic_idx}/{len(topic_folders)}: {topic_folder}")
            print(f"{'#'*70}\n")
            
            topic_path = os.path.join(base_dir, topic_folder)
            json_files = [f for f in os.listdir(topic_path) 
                         if f.endswith('.json') and not f.startswith('.')]
            
            topic_stats = {'successful': 0, 'failed': 0, 'qas': 0}
            
            for file_idx, filename in enumerate(json_files, 1):
                file_path = os.path.join(topic_path, filename)
                
                print(f"[{file_idx}/{len(json_files)}] {filename}")
                
                result = self.process_single_file(file_path)
                
                if result['success']:
                    print(f"   ✅ Schema stored | {result['qa_count']} Q&As stored")
                    stats['successful'] += 1
                    topic_stats['successful'] += 1
                    stats['total_qas'] += result['qa_count']
                    topic_stats['qas'] += result['qa_count']
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                    stats['failed'] += 1
                    topic_stats['failed'] += 1
                
                stats['total_files'] += 1
            
            stats['topics'][topic_folder] = topic_stats
            
            print(f"\n{'─'*70}")
            print(f"Topic: {topic_stats['successful']} successful, {topic_stats['failed']} failed, {topic_stats['qas']} Q&As")
            print(f"{'─'*70}")
        
        # Final summary
        self._print_summary(stats)
        
        return stats
    
    def _print_summary(self, stats: Dict[str, Any]):
        """Print comprehensive summary"""
        
        print(f"\n{'='*70}")
        print("🎉 UPLOAD COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Total files: {stats['total_files']}")
        print(f"✅ Successful: {stats['successful']}")
        print(f"❌ Failed: {stats['failed']}")
        print(f"📝 Total Q&As stored: {stats['total_qas']}")
        
        print(f"\n{'─'*70}")
        print("📂 BY TOPIC:")
        print(f"{'─'*70}")
        
        for topic, counts in stats['topics'].items():
            status = "✅" if counts['failed'] == 0 else "⚠️"
            print(f"{status} {topic}: {counts['successful']} docs, {counts['qas']} Q&As")
        
        print(f"\n{'─'*70}")
        print("🗄️  MONGODB COLLECTIONS:")
        print(f"{'─'*70}")
        print(f"📚 document_schemas: {self.schemas_collection.count_documents({})} documents")
        print(f"❓ document_qas: {self.qas_collection.count_documents({})} questions")
        print(f"{'='*70}\n")
    
    def get_document_types(self) -> List[str]:
        """Get all unique document types"""
        return self.qas_collection.distinct('document_type')
    
    def get_qas_by_document_type(self, document_type: str) -> List[Dict[str, Any]]:
        """
        Get all Q&As for a specific document type (for Streamlit UI)
        
        Args:
            document_type: Type of document
            
        Returns:
            List of Q&As sorted by category and order
        """
        qas = list(self.qas_collection.find(
            {'document_type': document_type}
        ).sort([
            ('category_order', ASCENDING),
            ('question_order', ASCENDING)
        ]))
        
        # Remove MongoDB _id for cleaner output
        for qa in qas:
            qa['_id'] = str(qa['_id'])
        
        return qas
    
    def update_answer(self, qa_id: str, answer: Any):
        """
        Update answer for a specific question
        
        Args:
            qa_id: MongoDB document ID
            answer: Answer value (string or list for structured_list)
        """
        from bson import ObjectId
        
        try:
            self.qas_collection.update_one(
                {'_id': ObjectId(qa_id)},
                {
                    '$set': {
                        'answer': answer,
                        '_runtime_metadata.last_updated': datetime.utcnow(),
                        '_runtime_metadata.answered': True
                    }
                }
            )
            return True
        except Exception as e:
            print(f"❌ Error updating answer: {e}")
            return False
    
    def get_schema_by_type(self, document_type: str) -> Dict[str, Any]:
        """Get full schema for a document type"""
        schema = self.schemas_collection.find_one({'document_type': document_type})
        if schema:
            schema['_id'] = str(schema['_id'])
        return schema
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        print("✅ MongoDB connection closed")


def main():
    """Main entry point for batch upload"""
    import sys
    
    print(f"\n{'='*70}")
    print("MONGODB INTEGRATION - BATCH UPLOAD")
    print(f"{'='*70}\n")
    
    # Configuration
    print("📋 Configuration:")
    print("─" * 70)
    
    # Get MongoDB connection string
    default_conn = "mongodb+srv://mohitturabit_db_user:yqAjPzqe5HuZ91Gl@cluster0.uzrpe8f.mongodb.net/"
    conn_string = os.getenv('MONGODB_CONNECTION_STRING', default_conn)
    
    print(f"MongoDB URI: {conn_string}")
    
    if conn_string == default_conn:
        print("💡 Using default local MongoDB")
        print("   To use remote MongoDB, set MONGODB_CONNECTION_STRING env variable")
    
    db_name = os.getenv('MONGODB_DATABASE', 'document_automation')
    print(f"Database: {db_name}")
    
    input_dir = os.getenv('INPUT_DIR', 'final_filtered_QAs')
    print(f"Input directory: {input_dir}")
    
    print("─" * 70)
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"\n❌ Directory not found: {input_dir}")
        print("   Please run add_answer_fields.py first to create the directory")
        sys.exit(1)
    
    # Count files
    total_files = 0
    for root, dirs, files in os.walk(input_dir):
        total_files += len([f for f in files if f.endswith('.json') and not f.startswith('.')])
    
    if total_files == 0:
        print(f"\n❌ No JSON files found in {input_dir}")
        sys.exit(1)
    
    print(f"\n📊 Found {total_files} files to upload")
    print(f"\n{'─'*70}\n")
    
    # Confirm
    confirm = input("Continue with upload to MongoDB? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("\n❌ Upload cancelled")
        sys.exit(0)
    
    # Initialize MongoDB integration
    try:
        mongo = MongoDBIntegration(
            connection_string=conn_string,
            database_name=db_name
        )
        
        # Process all files
        stats = mongo.process_directory(input_dir)
        
        # Close connection
        mongo.close()
        
        print(f"\n✨ Upload complete!")
        print(f"📊 {stats['successful']}/{stats['total_files']} files uploaded successfully")
        print(f"📝 {stats['total_qas']} questions ready for Streamlit UI\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure MongoDB is running and accessible")
        sys.exit(1)


if __name__ == "__main__":
    main()