import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from elsai_core.config.loggerConfig import setup_logger
class AzureDocumentIntelligence:
    """
    Class to handle document analysis using Azure Document Intelligence.
    """

    def __init__(self, file_path:str):
        self.logger = setup_logger()
        # Set up API key and endpoint
        self.key = os.environ["VISION_KEY"]
        self.endpoint = os.environ["VISION_ENDPOINT"]
        self.file_path = file_path
        # Initialize the Document Intelligence Client
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )

    def extract_text(self, pages: str = None) -> str:
        """
        Extracts text from a document with optional page selection.

        Args:
            
            pages (str, optional): Specific pages to analyze (e.g., "1,3"). Defaults to None.

        Returns:
            str: Extracted text content from the document.
        """

        self.logger.info("Starting text extraction from %s", self.file_path)
        try:

            with open(self.file_path, "rb") as f:
                self.logger.info("Opened file: %s", self.file_path)
                poller = self.client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=f,
                    content_type="application/octet-stream",
                    pages=pages
                )

            self.logger.info("Analysis started for %s. Waiting for result...", self.file_path)
            # Get the result of the analysis
            result = poller.result()
            self.logger.info("Analysis completed for %s", self.file_path)
            ocr_output = result.as_dict()
            self.logger.info("Text extraction from %s completed successfully.", self.file_path)
            return ocr_output['content']

        except Exception as e:
            self.logger.error("Error while extracting text from %s: %s", self.file_path, e)
            raise
