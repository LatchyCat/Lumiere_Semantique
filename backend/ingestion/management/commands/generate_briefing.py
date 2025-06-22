# In ingestion/management/commands/generate_briefing.py

from django.core.management.base import BaseCommand
from lumiere_core.services.ollama import search_index
from backend.lumiere_core.services.ollama_service import generate_text

class Command(BaseCommand):
    help = 'Generates a "Pre-flight Briefing" for a given query using a RAG pipeline.'

    def add_arguments(self, parser):
        parser.add_argument('repo_id', type=str, help="The ID of the repo (e.g., 'pallets_flask').")
        parser.add_argument('query', type=str, help='The user query or GitHub issue description.')
        parser.add_argument('--embedding_model', type=str, default='snowflake-arctic-embed2:latest', help='The Ollama model to use for embeddings.')
        # --- CHANGE 1: Add an argument for the generation model ---
        parser.add_argument('--generation_model', type=str, default='qwen3:4b', help='The Ollama model to use for text generation.')
        parser.add_argument('--k', type=int, default=7, help='Number of context chunks to retrieve.')

    def handle(self, *args, **options):
        repo_id = options['repo_id']
        query = options['query']
        embedding_model = options['embedding_model']
        generation_model = options['generation_model'] # <-- Get the new option
        k = options['k']

        self.stdout.write(self.style.NOTICE(f"Step 1: Retrieving context for query: '{query}'..."))

        index_path = f"{repo_id}_faiss.index"
        map_path = f"{repo_id}_id_map.json"

        try:
            context_chunks = search_index(
                query_text=query,
                model_name=embedding_model, # Use the embedding model here
                index_path=index_path,
                map_path=map_path,
                k=k
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to retrieve context: {e}"))
            return

        self.stdout.write(self.style.SUCCESS(f"✓ Retrieved {len(context_chunks)} context chunks."))

        context_string = ""
        for i, chunk in enumerate(context_chunks):
            context_string += f"--- Context Chunk {i+1} from file '{chunk['file_path']}' ---\n"
            context_string += chunk['text']
            context_string += "\n\n"

        prompt = f"""
        You are Lumière Sémantique, an expert AI programming assistant acting as a Principal Engineer.
        Your mission is to provide a "Pre-flight Briefing" for a developer about to work on a task.
        Analyze the user's query and the provided context from the codebase to generate your report.

        The report must be clear, concise, and structured in Markdown. It must include the following sections:
        1.  **Task Summary:** Briefly rephrase the user's request.
        2.  **Core Analysis:** Based on the provided context, explain how the system currently works in relation to the query. Synthesize information from the different context chunks.
        3.  **Key Files & Code:** Point out the most important files or functions from the context that the developer should focus on.
        4.  **Suggested Approach or Potential Challenges:** Offer a high-level plan or mention any potential issues you foresee.

        --- PROVIDED CONTEXT FROM THE CODEBASE ---
        {context_string}
        --- END OF CONTEXT ---

        USER'S QUERY: "{query}"

        Now, generate the Pre-flight Briefing.
        """

        self.stdout.write(self.style.NOTICE(f"\nStep 2: Sending context and query to the LLM ('{generation_model}') for generation..."))

        # --- CHANGE 2: Pass the generation model name to the function ---
        final_report = generate_text(prompt, model_name=generation_model)

        self.stdout.write(self.style.SUCCESS("\n--- LUMIÈRE SÉMANTIQUE: PRE-FLIGHT BRIEFING ---"))
        self.stdout.write(final_report)
