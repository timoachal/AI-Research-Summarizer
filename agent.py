import os
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
import io

load_dotenv()


class SummaryType(Enum):
    """Types of summaries that can be generated."""
    ABSTRACT = "abstract"  # 1-2 paragraph overview
    DETAILED = "detailed"  # 3-5 paragraph comprehensive summary
    BULLET_POINTS = "bullet_points"  # Key findings as bullet points
    TECHNICAL = "technical"  # Focus on methodology and results
    EXECUTIVE = "executive"  # High-level summary for non-experts


@dataclass
class PaperMetadata:
    """Metadata extracted from research paper."""
    title: Optional[str] = None
    authors: Optional[str] = None
    abstract: Optional[str] = None
    total_pages: int = 0
    extracted_text: str = ""


class QuotaExceededError(ValueError):
    """Raised when an anonymous user exceeds their token budget."""


class ResearchPaperSummarizer:
    """
    AI Agent for summarizing research papers using Groq LLM.
    Handles PDF processing, text extraction, and intelligent summarization.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the summarizer with Groq API credentials.
        
        Args:
            api_key: Groq API key. If not provided, uses GROQ_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not provided. Set it as an environment variable or pass it as argument."
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Strong general-purpose model
        self.conversation_history: List[Dict[str, str]] = []
        self.quota_preflight: Optional[Callable[[int], None]] = None
        self.usage_recorder: Optional[Callable[[int], None]] = None

    def configure_usage_hooks(
        self,
        quota_preflight: Optional[Callable[[int], None]] = None,
        usage_recorder: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Attach token quota hooks used by the app layer."""
        self.quota_preflight = quota_preflight
        self.usage_recorder = usage_recorder

    def _estimate_request_tokens(self, messages: List[Dict[str, str]], max_tokens: int) -> int:
        """Estimate total token cost conservatively before sending the request."""
        prompt_chars = 0
        for message in messages:
            prompt_chars += len(str(message.get("content", "")))
            prompt_chars += len(str(message.get("role", "")))
        prompt_tokens = max(1, prompt_chars // 4)
        return prompt_tokens + max_tokens

    def _generate_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float = 0.9,
    ) -> str:
        """Generate text with the Groq chat completions API."""
        estimated_tokens = self._estimate_request_tokens(messages, max_tokens)
        if self.quota_preflight is not None:
            self.quota_preflight(estimated_tokens)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
        )

        if self.usage_recorder is not None and completion.usage is not None:
            self.usage_recorder(completion.usage.total_tokens)

        content = completion.choices[0].message.content
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return str(content)

    def extract_text_from_pdf(self, pdf_file: Any) -> PaperMetadata:
        """
        Extract text and metadata from PDF file.
        
        Args:
            pdf_file: File object or path to PDF file.
            
        Returns:
            PaperMetadata object containing extracted information.
        """
        metadata = PaperMetadata()
        
        try:
            # Handle both file objects and file paths
            if isinstance(pdf_file, str):
                reader = PdfReader(pdf_file)
            else:
                reader = PdfReader(pdf_file)
            
            metadata.total_pages = len(reader.pages)
            
            # Extract text from all pages
            text_content = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_content.append(text)
            
            metadata.extracted_text = "\n".join(text_content)
            
            # Try to extract metadata from PDF
            if reader.metadata:
                metadata.title = reader.metadata.get("/Title", "Unknown Title")
                metadata.authors = reader.metadata.get("/Author", "Unknown Authors")
            
            # Try to extract abstract from first page (common location)
            first_page_text = text_content[0] if text_content else ""
            if "abstract" in first_page_text.lower():
                abstract_start = first_page_text.lower().find("abstract")
                abstract_end = first_page_text.lower().find("introduction", abstract_start)
                if abstract_end == -1:
                    abstract_end = len(first_page_text)
                metadata.abstract = first_page_text[abstract_start:abstract_end].strip()
            
            return metadata
            
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")

    def _create_summary_prompt(
        self, 
        text: str, 
        summary_type: SummaryType,
        max_tokens: int = 1000
    ) -> str:
        """
        Create an optimized prompt for the summarization task.
        
        Args:
            text: The research paper text to summarize.
            summary_type: Type of summary to generate.
            max_tokens: Maximum tokens for the summary.
            
        Returns:
            Formatted prompt string.
        """
        prompts = {
            SummaryType.ABSTRACT: f"""Provide a concise 1-2 paragraph abstract of this research paper. 
Focus on the main research question, methodology, and key findings.
Keep it under {max_tokens} tokens.

Paper:
{text[:4000]}""",
            
            SummaryType.DETAILED: f"""Provide a comprehensive 3-5 paragraph summary of this research paper.
Include:
1. Research question and motivation
2. Methodology and approach
3. Key findings and results
4. Implications and conclusions
5. Future work (if mentioned)

Keep it under {max_tokens} tokens.

Paper:
{text[:6000]}""",
            
            SummaryType.BULLET_POINTS: f"""Summarize this research paper as a structured list of key points.
Format as bullet points covering:
- Research Question
- Methodology
- Key Findings (3-5 points)
- Conclusions
- Limitations (if mentioned)

Paper:
{text[:5000]}""",
            
            SummaryType.TECHNICAL: f"""Provide a technical summary focusing on:
1. Problem Statement
2. Proposed Solution/Methodology
3. Experimental Setup
4. Results and Performance Metrics
5. Comparison with Related Work

Keep it technical and precise, under {max_tokens} tokens.

Paper:
{text[:6000]}""",
            
            SummaryType.EXECUTIVE: f"""Write an executive summary suitable for non-experts.
Explain in simple terms:
1. What problem does this research address?
2. Why is it important?
3. What did they do?
4. What did they find?
5. What does it mean for the real world?

Avoid jargon. Keep it under {max_tokens} tokens.

Paper:
{text[:5000]}"""
        }
        
        return prompts.get(summary_type, prompts[SummaryType.ABSTRACT])

    def summarize(
        self,
        text: str,
        summary_type: SummaryType = SummaryType.DETAILED,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a summary of the research paper using Groq LLM.
        
        Args:
            text: The research paper text to summarize.
            summary_type: Type of summary to generate.
            temperature: Temperature for LLM (0.0-1.0). Lower = more focused.
            
        Returns:
            Generated summary string.
        """
        if not text or len(text.strip()) < 100:
            raise ValueError("Text is too short to summarize (minimum 100 characters)")
        
        prompt = self._create_summary_prompt(text, summary_type)
        
        try:
            summary = self._generate_text(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=1500,
                top_p=0.9,
            )
            
            # Store in conversation history for context
            self.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": summary
            })
            
            return summary
            
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")

    def ask_followup_question(self, question: str) -> str:
        """
        Ask a follow-up question about the paper based on previous summaries.
        Maintains conversation context.
        
        Args:
            question: Follow-up question about the paper.
            
        Returns:
            Answer from the LLM.
        """
        if not self.conversation_history:
            raise ValueError("No previous summary context. Please summarize a paper first.")
        
        # Add context about maintaining conversation
        messages = self.conversation_history.copy()
        messages.append({
            "role": "user",
            "content": question
        })
        
        try:
            answer = self._generate_text(
                messages=messages,
                temperature=0.5,
                max_tokens=1000,
                top_p=0.9,
            )
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": question
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })
            
            return answer
            
        except Exception as e:
            raise Exception(f"Error answering follow-up question: {str(e)}")

    def extract_key_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extract key metrics and statistics from the paper.
        
        Args:
            text: The research paper text.
            
        Returns:
            Dictionary containing extracted metrics.
        """
        prompt = f"""Extract and list all key metrics, numbers, and statistics from this research paper.
Format as JSON with keys like 'accuracy', 'f1_score', 'dataset_size', 'improvement_percentage', etc.
Only include actual metrics found in the paper.

Paper excerpt:
{text[:3000]}

Return only valid JSON."""
        
        try:
            response_text = self._generate_text(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500,
            )
            
            # Try to parse JSON from response
            try:
                # Find JSON in response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            return {"raw_metrics": response_text}
            
        except Exception as e:
            return {"error": str(e)}

    def clear_history(self):
        """Clear conversation history for a new paper."""
        self.conversation_history = []


class PaperAnalyzer:
    """
    Advanced analyzer for comprehensive paper analysis.
    Combines multiple analysis techniques.
    """
    
    def __init__(self, summarizer: ResearchPaperSummarizer):
        """Initialize analyzer with a summarizer instance."""
        self.summarizer = summarizer
    
    def comprehensive_analysis(self, pdf_file: Any) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a research paper.
        
        Args:
            pdf_file: PDF file to analyze.
            
        Returns:
            Dictionary containing all analysis results.
        """
        # Extract text and metadata
        metadata = self.summarizer.extract_text_from_pdf(pdf_file)
        
        # Generate summaries
        abstract_summary = self.summarizer.summarize(
            metadata.extracted_text,
            SummaryType.ABSTRACT
        )
        
        detailed_summary = self.summarizer.summarize(
            metadata.extracted_text,
            SummaryType.DETAILED
        )
        
        bullet_summary = self.summarizer.summarize(
            metadata.extracted_text,
            SummaryType.BULLET_POINTS
        )
        
        # Extract metrics
        metrics = self.summarizer.extract_key_metrics(metadata.extracted_text)
        
        return {
            "metadata": {
                "title": metadata.title,
                "authors": metadata.authors,
                "pages": metadata.total_pages,
            },
            "abstract": abstract_summary,
            "detailed_summary": detailed_summary,
            "key_points": bullet_summary,
            "metrics": metrics,
        }
