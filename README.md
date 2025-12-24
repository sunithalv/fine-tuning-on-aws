# Fine-Tuning LLMs on AWS: Pharma RAG Application

## Project Objective

This project demonstrates a complete end-to-end machine learning pipeline for fine-tuning Large Language Models (LLMs) on pharmaceutical instruction data and deploying them on AWS infrastructure with Retrieval-Augmented Generation (RAG) capabilities. The system is designed to provide domain-specific responses for pharmaceutical queries by combining fine-tuned models with vector-based retrieval mechanisms.

### Key Goals:
- **Fine-tune** large language models on pharmaceutical instruction datasets using parameter-efficient methods (LoRA)
- **Deploy** models at scale using AWS SageMaker endpoints
- **Enable RAG** capabilities for contextual and accurate pharmaceutical information retrieval
- **Build interactive UI** for end-users to interact with the fine-tuned models
- **Serverless inference** using AWS Lambda for cost-effective inference

---

## Tools & Technologies Used

### Model Training & Fine-tuning
- **Transformers**: State-of-the-art NLP models (Hugging Face)
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Datasets**: Data loading and processing from CSV/Hugging Face Hub
- **BitsAndBytes**: Quantization support for memory-efficient training

### Vector & Retrieval Systems
- **FAISS**: Fast similarity search and vector indexing
- **LangChain**: Framework for building LLM applications with RAG capabilities
- **OpenAI Embeddings**: Text-embedding-3-small for semantic search
- **Google Generative AI Embeddings**: Alternative embedding models

### Cloud & Deployment
- **AWS SageMaker**: Model training, hosting, and inference endpoints
- **AWS Lambda**: Serverless inference function handler
- **AWS DynamoDB**: Logging and request tracking
- **AWS API Gateway**: REST API for Lambda integration
- **boto3**: AWS SDK for Python

### Frontend & Inference
- **Streamlit**: Interactive web UI for inference
- **FastAPI**: Backend API for RAG application
- **Requests**: HTTP client for API communication

### Data Processing
- **Pandas**: Data manipulation and analysis
- **PyArrow**: Columnar data format support

---

## Project Structure

```
fine-tuning-on-aws/
├── scripts/
│   └── train.py                      # Training script with LoRA fine-tuning
├── finetuning_experiment/
│   └── experiment.ipynb             # Experimentation and exploration notebook
├── deployment_of_model.ipynb        # Model deployment workflow
├── estimator_launcher.ipynb         # SageMaker job launcher notebook
├── inference_app.py                 # Streamlit inference UI
├── rag_app_backend.py               # RAG backend with FAISS & LangChain
├── rag_app_ui.py                    # RAG application UI (current)
├── rag_app_ui_deprecated.py         # Legacy UI version
├── lambda_function.py               # AWS Lambda handler for serverless inference
├── pharma_instruction_data.csv      # Pharmaceutical instruction dataset
├── requirements.txt                 # Production dependencies
├── requirements_inference.txt       # Inference-only dependencies
└── README.md                        # This file
```

---

## Deployment Strategies

### 1. **SageMaker Endpoint Deployment** (Primary)

**Use Case**: Production inference with auto-scaling capabilities

**Workflow**:
- Train model using SageMaker Training Jobs (`scripts/train.py`)
- Deploy to SageMaker Hosting endpoint
- Invoke via `boto3` SageMaker Runtime
- Enable auto-scaling based on request volume

**Advantages**:
- Managed inference infrastructure
- Built-in monitoring and logging
- Model versioning support
- A/B testing capabilities

**Files**:
- `deployment_of_model.ipynb` - Deployment steps
- `estimator_launcher.ipynb` - Job submission
- `lambda_function.py` - Inference handler

---

### 2. **AWS Lambda + API Gateway** (Serverless)

**Use Case**: Cost-effective, event-driven inference

**Architecture**:
```
API Gateway → Lambda Function → SageMaker Endpoint / DynamoDB Logging
```

**Workflow**:
1. User sends request to API Gateway
2. Lambda function receives event payload
3. Parses input and invokes SageMaker endpoint
4. Logs request/response to DynamoDB
5. Returns response in Lambda Proxy format

**Advantages**:
- Pay-per-invocation pricing
- Automatic scaling (no cold start concerns for low volume)
- Event-driven architecture
- Easy integration with other AWS services

**Configuration**:
- Set environment variables: `SAGEMAKER_ENDPOINT`, `LOG_TABLE`
- API payload format: `{"inputs": "prompt text"}`

---

### 3. **Streamlit Web Application**

**Use Case**: Interactive UI for end-users and stakeholders

**Architecture**:
```
Streamlit App → API Gateway → Lambda/SageMaker Endpoint
```

**Features**:
- Text input for queries
- RAG capability with FAISS vector store
- Response streaming
- Debug information display (raw JSON output)

**Files**:
- `inference_app.py` - Basic inference interface
- `rag_app_ui.py` - RAG-enabled application
- `rag_app_backend.py` - Backend logic with embeddings

**Environment Variables**:
- `API_URL` - API Gateway endpoint
- `API_KEY` - Optional API authentication
- `OPENAI_API_KEY` - For OpenAI embeddings

---

### 4. **RAG (Retrieval-Augmented Generation)**

**Use Case**: Enhanced pharmaceutical knowledge retrieval

**Components**:
- **Vector Store**: FAISS for semantic similarity search
- **Embeddings**: OpenAI text-embedding-3-small
- **Documents**: Pharmaceutical knowledge base (drugs, medications, etc.)
- **LLM Integration**: Chains using LangChain for context-aware responses

**Workflow**:
```
User Query → Embed & Search FAISS → Retrieve Context → Feed to LLM → Generate Response
```

**Implementation**:
- Vector store initialized with pharmaceutical documents
- Similarity search on user queries
- Context passed to model for grounded responses

---

## Getting Started

### Prerequisites
- Python 3.11+
- AWS Account with SageMaker, Lambda, and DynamoDB access
- OpenAI API key (for embeddings)
- Virtual environment (recommended)

### Installation

1. **Create virtual environment**:
   ```bash
   python -m venv env
   source env/Scripts/activate  # Windows: env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (`.env` file):
   ```
   OPENAI_API_KEY=your_openai_key
   API_URL=your_api_gateway_url
   API_KEY=your_api_key
   ```

### Training

```bash
python scripts/train.py \
    --model_id meta-llama/Llama-2-7b-hf \
    --epochs 3 \
    --per_device_train_batch_size 8 \
    --lr 2e-4 \
    --train_data ./
```

### Running Inference

**Streamlit App**:
```bash
streamlit run inference_app.py
```

**RAG Application**:
```bash
streamlit run rag_app_ui.py
```

---

## Key Features

✅ **Parameter-Efficient Fine-Tuning**: LoRA for reduced memory footprint  
✅ **Scalable Inference**: SageMaker endpoints with auto-scaling  
✅ **Serverless Deployment**: Lambda + API Gateway integration  
✅ **RAG Capabilities**: FAISS vector search with LangChain  
✅ **Production Logging**: DynamoDB integration for tracking  
✅ **Interactive UI**: Streamlit for easy access  
✅ **Domain-Specific**: Pharmaceutical instruction data  

---

## Configuration & AWS Resources

### Required AWS IAM Permissions:
- SageMaker training and hosting
- Lambda execution
- DynamoDB read/write
- API Gateway invoke
- S3 access for data and models

### Environment Variables:
| Variable | Purpose |
|----------|---------|
| `SAGEMAKER_ENDPOINT` | Deployed model endpoint name |
| `LOG_TABLE` | DynamoDB table for logging |
| `API_URL` | API Gateway endpoint URL |
| `API_KEY` | Optional authentication key |
| `OPENAI_API_KEY` | OpenAI embeddings API key |

---

## Monitoring & Logging

- **CloudWatch**: SageMaker job and endpoint logs
- **DynamoDB**: Request/response logging via Lambda
- **Streamlit**: Debug console for local development

---

## Future Enhancements

- [ ] Multi-model ensemble support
- [ ] Fine-grained RAG with metadata filtering
- [ ] Advanced caching mechanisms
- [ ] A/B testing framework
- [ ] Cost optimization dashboard

---

## License

This project is provided as-is for educational and research purposes.

---

## Contact & Support

For questions or issues, please refer to the AWS SageMaker documentation and the respective library documentation for Transformers, LangChain, and FAISS.
