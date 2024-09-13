from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import itertools
import warnings
import re
import pandas as pd
import transformers
import torch


class RAG:
    def __init__(self,
                 pinecone_api_key: str
                 ):
        """
        Initialises a RAG instance. The RAG uses Pinecone as the database and paraphrase-multilingual-MiniLM-L12-v2
        as the embedding model.
        Setup the Pinecone information for vector upsertion and database query.
        :param pinecone_api_key: The private Pinecone API key.
        """
        self.pinecone_api_key = pinecone_api_key
        pinecone = Pinecone(pinecone_api_key)
        self.pinecone = pinecone

    def create_index(self,
                     pinecone_index_name: str,
                     vector_dims: int = 384,
                     vector_metric: str = "cosine",
                     cloud_type: str = "aws",
                     region: str = "us-east-1"
                     ):
        """
        Create an pinecone index
        :param vector_dims: The number of dimensions of the vectors for the database. Defaults to 384 for
        paraphrase-multilingual-MiniLM-L12-v2.
        :param vector_metric: Similarity-metric to use for query's. Defaults to "cosine" for the
        paraphrase-multilingual-MiniLM-L12-v2.
        :param cloud_type: Specify the Pinecone-index cloud type. Defaults to 'aws' for the free version.
        :param region: Specify the Pinecone-server region. Defaults to 'us-east-1' for the free version.
        :param pinecone_index_name: Name of the pinecone index to create.
        :return: None
        """
        self.pinecone.create_index(
            name=pinecone_index_name,
            dimension=vector_dims,
            metric=vector_metric,
            spec=ServerlessSpec(
                cloud=cloud_type,
                region=region
            )
        )

    def __chunks(self, iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def upsert_vectors_to_index(self, pinecone_index_name: str, vectors: list):
        """
        Upsert vectors to pinecone index. The vectors should be a list of dictionaries that is suitable for pinecone
        upsertion (for reference: https://docs.pinecone.io/guides/data/upsert-data).
        :param pinecone_index_name: Name of the pinecone index.
        :param vectors: List of dictionaries containing the vectors to be upserted.
        :return: None
        """
        index = self.pinecone.Index(pinecone_index_name)
        for vec_chunks in self.__chunks(vectors):
            index.upsert(vectors=vec_chunks)

    def __get_chunk_content(self, input_string, filepath="Chunks.csv") -> str:
        """
        Get chunk content basen on the Pinecone-Vector-id.
        :param input_string: Pinecone-vector-id.
        :param filepath: Path to the Chunks.csv file.
        :return: Content of chunk as string.
        """
        df = pd.read_csv(filepath, sep=";",
                         encoding="latin1")
        pattern = r'mh(\d+)ch(\d+)'
        match = re.search(pattern, input_string)
        if match:
            id = int(match.group(1))
            chunk = int(match.group(2))
            content = df[(df['file'] == id) & (df['chunk'] == chunk)]['text'].tolist()[0]
            return content
        else:
            warnings.warn("The input string does not match the pattern!", RuntimeWarning)

    def __dict_to_string(self, dictionary: dict) -> str:
        """
        Transforms key-value pairs into a string.
        :param dictionary: dictionary
        :return: string
        """
        _res = ""
        for key, value in dictionary.items():
            _res += f"{key}: {value}\n"
        return _res

    def __format_result(self, l: list, chunk_path="Chunks.csv") -> str:
        """
        Format the result of a query into a string.
        :param l: Content list.
        :return: String.
        """
        result = ""
        for i, qr in enumerate(l):
            content = self.__get_chunk_content(qr['id'], filepath=chunk_path)
            metadata = self.__dict_to_string(qr['metadata'])
            result += (f"#### Abschnitt {i + 1} ####\n"
                       f"### Metadata ###\n"
                       f"{metadata}\n"
                       f"### Inhalt ###\n"
                       f"{content}\n\n")
        return result

    def query(self,
              question: str,
              pinecone_index_name: str,
              top_k: int = 13,
              chunk_csv_path="Chunks.csv",
              generate_answer_using_llama3=False
              ) -> str:
        """
        Identifies the top_k most relevant chunks based on a question and formates the output.
        Returns a string with the top_k chunks and correlating metadata ordered by their similarity to the question.
        :param generate_answer_using_llama3: Generates an answer based on the retrieved IInformation.
        [Notice!] This requires fitting hardware and a hugginface api Token ('HF_TOKEN') as an environment secret.
        :param chunk_csv_path: PAth to a csv file containing the chunks. Each chunk has a modulhandbook number, a chunk
        number and the content of the chunk. Default is 'Chunks.csv'.
        :param question: The question that the LLM should answer.
        :param pinecone_index_name: Name of the pinecone index that contains the embeddings of the chunks.
        :param top_k: Number of chunks to return. Default is 13 (fitted to Llama3-8B-Instruct).
        :return: If generate_answer_using_llama3=False: A formatted string with the content of chunks and their metadat.
        This string is ready to be presented to a LLM for the question-answering step.
        If generate_answer_using_llama3=True: It returns the answer of the LLM for the asked question.
        """
        if question.strip() != "":
            # Retrieve relevant chunks
            # Convert question into vector
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            question_vector = model.encode(question).tolist()

            # Connect to Pinecone Database
            index = self.pinecone.Index(pinecone_index_name)
            query_results = index.query(
                vector=question_vector,
                top_k=top_k,
                include_metadata=True
            )

            result = self.__format_result(query_results['matches'], chunk_path=chunk_csv_path)

            if not generate_answer_using_llama3:
                return result
            else:
                # Use a pipline to talk to llama3
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_id,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )
                # Instruction text for the llm
                instruction = ("Beantworte die Folgende Frage präzise anhand der Abschnitte! Wenn die Abschnitte keine"
                               "Informationen bezüglich der Frage enthalten, dann antworte mit: 'Tut mir Leid, leider"
                               "kann ich keine Informationen dazu finden.' Erwähne nie, dass du Abschnitte durchsuchst!"
                               )

                # Message for the LLM
                messages = [
                    {
                        "role": "system",
                        "content": instruction
                    },
                    {
                        "role": "user",
                        "content": f"Frage: {question}\nAbschnitte: {result}"
                    },
                ]
                # define a stop-token for generation, otherwise the LLM tries to force fill the max_new_tokens length.
                terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                # generate an answer
                outputs = pipeline(
                    messages,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.14,
                    top_k=49
                )
                res = outputs[0]["generated_text"][-1]
                # clear gpu cache to prevent running out of memory after multiple queries
                torch.cuda.empty_cache()

                return res

        else:
            ValueError("A non valid question was provided!")

