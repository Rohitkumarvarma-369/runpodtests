import json
import numpy as np
import re
import argparse
import sys
from typing import List, Dict, Any, Tuple
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # To avoid warnings

# Import necessary libraries
try:
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "sentence-transformers", "transformers", "torch>=2.0.0", 
                          "accelerate", "scikit-learn"])
    
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from sklearn.metrics.pairwise import cosine_similarity

class FinancialScenariosRAG:
    def __init__(self, 
                 llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 scenarios_file: str = None):
        """
        Initialize the RAG system with scenario data and models.
        
        Args:
            llm_model_name: Name of the language model from Hugging Face
            embedding_model: Name of sentence embedding model
            scenarios_file: Path to JSON file containing scenarios data
        """
        # Load scenarios data
        self.scenarios = self._load_scenarios(scenarios_file)
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Create vector index
        print("Creating vector index...")
        self._create_vector_index()
        
        # Initialize LLM with Transformers
        print(f"Loading LLM from Hugging Face: {llm_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Create a standard pipeline without 4-bit quantization
            self.generator = pipeline(
                "text-generation",
                model=llm_model_name,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
                
            print("LLM loaded successfully!")
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Continuing without LLM capabilities. Only retrieval will be available.")
            self.generator = None

        print("RAG system initialized and ready!")

    def _load_scenarios(self, scenarios_file: str = None) -> List[Dict[str, str]]:
        """
        Load scenarios from a JSON file or use default hardcoded scenarios if no file specified
        
        Args:
            scenarios_file: Path to JSON file containing scenarios data
            
        Returns:
            List of scenario dictionaries
        """
        if scenarios_file and os.path.exists(scenarios_file):
            print(f"Loading scenarios from {scenarios_file}")
            try:
                with open(scenarios_file, 'r') as f:
                    data = json.load(f)
                    if "scenarios" in data:
                        return data["scenarios"]
                    else:
                        print("Warning: JSON file doesn't contain 'scenarios' key. Using default scenarios.")
            except Exception as e:
                print(f"Error loading scenarios from file: {e}")
                print("Using default scenarios instead.")
        else:
            if scenarios_file:
                print(f"Scenarios file {scenarios_file} not found. Using default scenarios.")
        
        # Fallback to default hardcoded scenarios
        print("Using default hardcoded scenarios")
        scenarios_json = """
        {
          "scenarios": [
            {
              "id": "scenario-001",
              "description": "Participant with zero defined contributions balance"
            },
            {
              "id": "scenario-002",
              "description": "Participant under age 25 with no phone number linked to account"
            },
            {
              "id": "scenario-003",
              "description": "Participants who made more than 3 withdrawals in the last quarter"
            },
            {
              "id": "scenario-004",
              "description": "Accounts that haven't been accessed online in over 12 months"
            },
            {
              "id": "scenario-005",
              "description": "Participants eligible for catch-up contributions but not utilizing them"
            },
            {
              "id": "scenario-006",
              "description": "Participants approaching retirement age (60+) with aggressive portfolio allocations"
            },
            {
              "id": "scenario-007",
              "description": "Accounts with beneficiary information missing or incomplete"
            },
            {
              "id": "scenario-008",
              "description": "Participants who have opted out of electronic communications"
            },
            {
              "id": "scenario-009",
              "description": "Accounts with international mailing addresses"
            },
            {
              "id": "scenario-010",
              "description": "Participants with contributions exceeding annual IRS limits"
            },
            {
              "id": "scenario-011",
              "description": "Accounts with pending compliance review flags"
            },
            {
              "id": "scenario-012",
              "description": "Participants who have changed their investment allocation more than 5 times in the past month"
            },
            {
              "id": "scenario-013",
              "description": "Accounts with uninvested cash balance exceeding $10,000 for more than 30 days"
            },
            {
              "id": "scenario-014",
              "description": "Participants who have requested paper statements despite online account access"
            },
            {
              "id": "scenario-015",
              "description": "Accounts with negative year-to-date returns exceeding 15%"
            },
            {
              "id": "scenario-016",
              "description": "Participants who have not updated their personal information in over 5 years"
            },
            {
              "id": "scenario-017",
              "description": "Accounts with outstanding loan balances past due over 90 days"
            },
            {
              "id": "scenario-018",
              "description": "Participants with multiple active retirement accounts under the same employer"
            },
            {
              "id": "scenario-019",
              "description": "Accounts flagged for potential identity theft or fraud"
            },
            {
              "id": "scenario-020",
              "description": "Participants who have recently become eligible for Required Minimum Distributions"
            },
            {
              "id": "scenario-021",
              "description": "Accounts with recurring deposit failures in the last quarter"
            },
            {
              "id": "scenario-022",
              "description": "Participants with 100% allocation to a single investment option"
            },
            {
              "id": "scenario-023",
              "description": "Accounts created but never funded after 60 days"
            },
            {
              "id": "scenario-024",
              "description": "Participants who have contacted customer service more than 5 times in the past month"
            },
            {
              "id": "scenario-025",
              "description": "Accounts with recent large withdrawals (>$50,000) in a single transaction"
            },
            {
              "id": "scenario-026",
              "description": "Participants who have opted for automatic contribution increases"
            },
            {
              "id": "scenario-027",
              "description": "Accounts with pending rollover transactions older than 30 days"
            },
            {
              "id": "scenario-028",
              "description": "Participants who have not designated a trusted contact person"
            },
            {
              "id": "scenario-029",
              "description": "Accounts with excessive trading activity triggering monitoring alerts"
            },
            {
              "id": "scenario-030",
              "description": "Participants who are military veterans eligible for special benefits"
            },
            {
              "id": "scenario-031",
              "description": "Accounts with returned mail or undeliverable address flags"
            },
            {
              "id": "scenario-032",
              "description": "Participants who have recently changed their marital status"
            },
            {
              "id": "scenario-033",
              "description": "Accounts with duplicate tax identification numbers"
            },
            {
              "id": "scenario-034",
              "description": "Participants who have enabled multi-factor authentication"
            },
            {
              "id": "scenario-035",
              "description": "Accounts with target date funds not aligned with expected retirement date"
            },
            {
              "id": "scenario-036",
              "description": "Participants who have taken coronavirus-related distributions"
            },
            {
              "id": "scenario-037",
              "description": "Accounts with recent power of attorney designations"
            },
            {
              "id": "scenario-038",
              "description": "Participants who have opted into auto-rebalancing but haven't had a rebalance in 12+ months"
            },
            {
              "id": "scenario-039",
              "description": "Accounts belonging to participants over age 70 without RMD withdrawals"
            },
            {
              "id": "scenario-040",
              "description": "Participants who have recently relocated to a different state"
            },
            {
              "id": "scenario-041",
              "description": "Accounts with hardship withdrawal requests in the last 90 days"
            },
            {
              "id": "scenario-042",
              "description": "Participants with employer matching contributions not being fully utilized"
            },
            {
              "id": "scenario-043",
              "description": "Accounts with uncashed distribution checks older than 180 days"
            },
            {
              "id": "scenario-044",
              "description": "Participants who have recently terminated employment but haven't initiated account transfer"
            },
            {
              "id": "scenario-045",
              "description": "Accounts with incomplete tax withholding information"
            },
            {
              "id": "scenario-046",
              "description": "Participants with non-spouse beneficiaries for IRA accounts"
            },
            {
              "id": "scenario-047",
              "description": "Accounts with state unclaimed property reporting flags"
            },
            {
              "id": "scenario-048",
              "description": "Participants who have reached lifetime maximum for loans"
            },
            {
              "id": "scenario-049",
              "description": "Accounts with recent name changes requiring additional verification"
            },
            {
              "id": "scenario-050",
              "description": "Participants who have recently enrolled in automatic investment advice services"
            },
            {
              "id": "scenario-051",
              "description": "Accounts with foreign tax reporting requirements"
            },
            {
              "id": "scenario-052",
              "description": "Participants with backdoor Roth IRA conversions in current tax year"
            },
            {
              "id": "scenario-053",
              "description": "Accounts subject to QDRO (Qualified Domestic Relations Order) proceedings"
            },
            {
              "id": "scenario-054",
              "description": "Participants who have opted out of default investment options"
            },
            {
              "id": "scenario-055",
              "description": "Accounts with recent large transfers between investment options (>$100,000)"
            },
            {
              "id": "scenario-056",
              "description": "Participants with multiple failed login attempts in the past week"
            },
            {
              "id": "scenario-057",
              "description": "Accounts with dormancy fees assessed in the last quarter"
            },
            {
              "id": "scenario-058",
              "description": "Participants who have recently downloaded tax documents"
            },
            {
              "id": "scenario-059",
              "description": "Accounts with pending death benefit claims"
            },
            {
              "id": "scenario-060",
              "description": "Participants with recent changes to contribution percentages"
            }
          ]
        }
        """
        # Parse the JSON string
        try:
            data = json.loads(scenarios_json)
            return data["scenarios"]
        except json.JSONDecodeError:
            print("Error parsing scenarios JSON")
            return []

    def _create_vector_index(self):
        """Create a vector index for similarity search"""
        # Create text representations for embedding
        self.scenario_texts = [
            f"{scenario['id']}: {scenario['description']}" 
            for scenario in self.scenarios
        ]
        
        # Generate embeddings
        self.embeddings = self.embedding_model.encode(self.scenario_texts)
        
        # Normalize embeddings for cosine similarity
        self.normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Create a lookup from ID to scenario
        self.id_to_scenario = {scenario["id"]: scenario for scenario in self.scenarios}

    def retrieve_by_id(self, scenario_id: str) -> List[Dict[str, str]]:
        """
        Retrieve a scenario by its ID
        
        Args:
            scenario_id: ID of the scenario (e.g., "scenario-045" or just "45")
            
        Returns:
            List of matching scenarios
        """
        # Format the scenario ID properly
        if re.match(r'^\d+$', scenario_id):
            # If it's just a number, format it as scenario-NNN
            scenario_id = f"scenario-{int(scenario_id):03d}"
        
        # Check if this ID exists
        if scenario_id in self.id_to_scenario:
            return [self.id_to_scenario[scenario_id]]
        return []

    def retrieve_by_similarity(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        Retrieve scenarios by semantic similarity
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of matching scenarios
        """
        # Embed the query
        query_embedding = self.embedding_model.encode([query])
        
        # Compute cosine similarity between query and all scenarios
        similarities = cosine_similarity(query_embedding, self.normalized_embeddings)[0]
        
        # Get indices of top-k most similar scenarios
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return the matching scenarios
        results = [self.scenarios[idx] for idx in top_indices]
        return results

    def answer_query(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Answer a query about scenarios
        
        Args:
            query: The user's question
            
        Returns:
            Tuple of (answer text, relevant scenarios)
        """
        # Check if the query is asking for a specific scenario by ID
        scenario_id_match = re.search(r'scenario\s*[-#]?\s*(\d+)', query, re.IGNORECASE)
        exact_number_match = re.search(r'^(\d+)$', query)
        
        retrieved_scenarios = []
        
        # If it's a direct ID query
        if scenario_id_match:
            scenario_id = scenario_id_match.group(1)
            retrieved_scenarios = self.retrieve_by_id(scenario_id)
        elif exact_number_match:
            scenario_id = exact_number_match.group(1)
            retrieved_scenarios = self.retrieve_by_id(scenario_id)
        
        # If no exact match by ID or ID not found, try semantic search
        if not retrieved_scenarios:
            retrieved_scenarios = self.retrieve_by_similarity(query)
        
        # If no LLM available, just return the retrieved information
        if self.generator is None:
            retrieved_info = "\n".join([f"{s['id']}: {s['description']}" for s in retrieved_scenarios])
            return f"Retrieved scenarios:\n{retrieved_info}", retrieved_scenarios
        
        # Prepare context for LLM
        context = "Financial scenarios information:\n"
        context += "\n".join([f"{s['id']}: {s['description']}" for s in retrieved_scenarios])
        
        # Prepare prompt
        prompt = f"""<|im_start|>system
You are a helpful financial advisor assistant that provides information about different financial scenarios in retirement accounts.

CONTEXT INFORMATION:
{context}

Based only on the context information provided above, please answer the user's query clearly and concisely.
If the context doesn't contain relevant information, say "I don't have specific information about that scenario."
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        
        # Generate response with Transformers pipeline
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=500,
                do_sample=True, 
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = outputs[0]['generated_text']
            
            # Extract only the assistant's response (after last <|im_start|>assistant)
            assistant_text = generated_text.split("<|im_start|>assistant")[-1]
            
            # Clean up any remaining tokens or delimiters
            assistant_text = assistant_text.replace("<|im_end|>", "").strip()
            
            return assistant_text, retrieved_scenarios
            
        except Exception as e:
            print(f"Error generating response: {e}")
            retrieved_info = "\n".join([f"{s['id']}: {s['description']}" for s in retrieved_scenarios])
            return f"Retrieved scenarios (LLM error):\n{retrieved_info}", retrieved_scenarios

def main():
    """Main function to run the RAG system interactively"""
    parser = argparse.ArgumentParser(description="Financial Scenarios RAG")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Name of HuggingFace model to use")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Path to JSON file containing scenarios data")
    args = parser.parse_args()
    
    try:
        print("\nInitializing Financial Scenarios RAG System...")
        print("This may take a few minutes as models are downloaded...")
        rag_system = FinancialScenariosRAG(llm_model_name=args.model, scenarios_file=args.scenarios)
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        sys.exit(1)
    
    print("\n===== Financial Scenarios RAG System =====")
    print("Type your questions about financial scenarios (e.g., 'What is scenario 45?')")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
                
            if not query:
                continue
                
            print("\nProcessing query...")
            answer, scenarios = rag_system.answer_query(query)
            
            print("\n----- Answer -----")
            print(answer)
            print("\n----- Retrieved Scenarios -----")
            for scenario in scenarios:
                print(f"{scenario['id']}: {scenario['description']}")
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
