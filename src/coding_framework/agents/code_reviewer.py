"""
Code Reviewer Agent for analyzing and improving code quality.

This agent specializes in reviewing code for correctness, style, performance,
security vulnerabilities, and adherence to best practices.
"""

import re
import time
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentResponse


class CodeReview(BaseModel):
    """Structured code review result."""
    
    overall_score: float = Field(..., description="Overall code quality score (0-100)")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Found issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    strengths: List[str] = Field(default_factory=list, description="Code strengths")
    security_concerns: List[str] = Field(default_factory=list, description="Security issues")
    performance_notes: List[str] = Field(default_factory=list, description="Performance observations")


class CodeReviewerAgent(BaseAgent):
    """
    Specialized agent for code review and quality analysis.
    
    Reviews code for correctness, style, performance, security,
    and adherence to best practices with detailed feedback.
    """
    
    @property
    def agent_type(self) -> str:
        """Return agent type identifier."""
        return "code_reviewer"
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for code review."""
        return """You are a Senior Code Reviewer with expertise across multiple programming languages. Your role is to:

1. **Analyze Code Quality**: Review code for correctness, readability, and maintainability
2. **Identify Issues**: Find bugs, potential errors, and areas for improvement
3. **Security Assessment**: Identify potential security vulnerabilities
4. **Performance Review**: Assess performance implications and suggest optimizations
5. **Best Practices**: Ensure adherence to language-specific best practices
6. **Constructive Feedback**: Provide helpful, actionable suggestions

**Review Categories:**
- **Correctness**: Logic errors, edge cases, potential bugs
- **Style**: Code formatting, naming conventions, structure
- **Performance**: Efficiency, algorithmic complexity, resource usage
- **Security**: Vulnerabilities, input validation, secure coding practices
- **Maintainability**: Code organization, documentation, testability
- **Best Practices**: Language idioms, design patterns, conventions

**Review Format:**
Provide your review in the following structure:

**Overall Assessment:** [Brief summary and score 0-100]

**Issues Found:**
- [Issue 1: Description, Severity, Location]
- [Issue 2: Description, Severity, Location]

**Suggestions:**
- [Specific actionable improvements]

**Strengths:**
- [What the code does well]

**Security Concerns:**
- [Any security-related observations]

**Performance Notes:**
- [Performance-related observations]

Be constructive, specific, and provide examples where helpful."""
    
    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Review code and provide detailed analysis.
        
        Args:
            request: Code to review (can include problem context)
            context: Additional context (language, focus areas, etc.)
            **kwargs: Review parameters
            
        Returns:
            Agent response with detailed code review
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting code review", request_length=len(request))
            
            # Extract review parameters
            focus_areas = kwargs.get("focus_areas") or (context or {}).get("focus_areas", ["all"])
            severity_filter = kwargs.get("severity_filter", "all")  # low, medium, high, all
            include_suggestions = kwargs.get("include_suggestions", True)
            
            # Update state
            self.update_state("last_review_request", request)
            self.update_state("focus_areas", focus_areas)
            
            # Extract code from request if it contains context
            code_info = self._extract_code_and_context(request)
            
            # Build enhanced review request
            review_request = self._build_review_request(
                code_info["code"],
                code_info["context"],
                focus_areas,
                include_suggestions,
            )
            
            # Prepare messages for LLM
            messages = self._build_messages(review_request, context)
            
            # Perform review using LLM
            response = await self._call_llm(
                messages,
                temperature=kwargs.get("temperature", 0.3),  # Lower temp for consistency
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            
            # Parse and structure the review
            structured_review = self._parse_review_response(response)
            
            # Filter issues by severity if requested
            if severity_filter != "all":
                structured_review["issues"] = self._filter_issues_by_severity(
                    structured_review["issues"], severity_filter
                )
            
            # Create response metadata
            metadata = {
                "focus_areas": focus_areas,
                "severity_filter": severity_filter,
                "code_length": len(code_info["code"]),
                "issues_found": len(structured_review["issues"]),
                "overall_score": structured_review.get("overall_score", 0),
                "review_categories": self._categorize_issues(structured_review["issues"]),
                "language": self._detect_language(code_info["code"]),
            }
            
            # Add to conversation history
            self.add_to_history(HumanMessage(content=request))
            self.add_to_history(HumanMessage(content=response))
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                "Code review completed",
                execution_time=execution_time,
                **metadata,
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                content=response,
                metadata=metadata,
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.logger.error(
                "Code review failed",
                error=str(e),
                execution_time=execution_time,
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                content="",
                metadata={"error_type": type(e).__name__},
                execution_time=execution_time,
                error=str(e),
            )
    
    def _extract_code_and_context(self, request: str) -> Dict[str, str]:
        """
        Extract code and context from review request.
        
        Args:
            request: Raw review request
            
        Returns:
            Dictionary with extracted code and context
        """
        # Look for code blocks
        code_block_pattern = r"```(?:\w+\n)?(.*?)```"
        code_matches = re.findall(code_block_pattern, request, re.DOTALL)
        
        if code_matches:
            # Take the largest code block
            code = max(code_matches, key=len).strip()
            
            # Context is everything else
            context = re.sub(code_block_pattern, "[CODE_BLOCK_REMOVED]", request, flags=re.DOTALL).strip()
        else:
            # Assume entire request is code if no code blocks
            code = request.strip()
            context = ""
        
        return {"code": code, "context": context}
    
    def _build_review_request(
        self,
        code: str,
        context: str,
        focus_areas: List[str],
        include_suggestions: bool,
    ) -> str:
        """
        Build comprehensive review request.
        
        Args:
            code: Code to review
            context: Additional context
            focus_areas: Areas to focus on
            include_suggestions: Whether to include improvement suggestions
            
        Returns:
            Enhanced review request
        """
        request_parts = []
        
        if context:
            request_parts.append(f"**Context:** {context}")
        
        request_parts.append(f"**Code to Review:**\n```\n{code}\n```")
        
        if focus_areas and focus_areas != ["all"]:
            focus_str = ", ".join(focus_areas)
            request_parts.append(f"**Focus Areas:** {focus_str}")
        
        if include_suggestions:
            request_parts.append("**Requirements:** Include specific improvement suggestions with examples where helpful.")
        
        request_parts.append("**Please provide a comprehensive code review following the specified format.**")
        
        return "\n\n".join(request_parts)
    
    def _parse_review_response(self, response: str) -> Dict[str, Any]:
        """
        Parse structured review from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Structured review dictionary
        """
        structured_review = {
            "overall_score": 0,
            "issues": [],
            "suggestions": [],
            "strengths": [],
            "security_concerns": [],
            "performance_notes": [],
        }
        
        # Extract overall score
        score_match = re.search(r"score.*?(\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if score_match:
            try:
                structured_review["overall_score"] = float(score_match.group(1))
            except ValueError:
                pass
        
        # Extract sections using regex patterns
        sections = {
            "issues": [r"Issues Found:(.*?)(?=\*\*|$)", r"Problems:(.*?)(?=\*\*|$)"],
            "suggestions": [r"Suggestions:(.*?)(?=\*\*|$)", r"Improvements:(.*?)(?=\*\*|$)"],
            "strengths": [r"Strengths:(.*?)(?=\*\*|$)", r"Good points:(.*?)(?=\*\*|$)"],
            "security_concerns": [r"Security Concerns:(.*?)(?=\*\*|$)", r"Security:(.*?)(?=\*\*|$)"],
            "performance_notes": [r"Performance Notes:(.*?)(?=\*\*|$)", r"Performance:(.*?)(?=\*\*|$)"],
        }
        
        for section_key, patterns in sections.items():
            items = []
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    # Extract bullet points
                    items = [
                        item.strip().lstrip("- ").lstrip("* ")
                        for item in content.split("\n")
                        if item.strip() and (item.strip().startswith("-") or item.strip().startswith("*"))
                    ]
                    break
            
            if section_key == "issues":
                # Parse issues with severity
                structured_review[section_key] = self._parse_issues(items)
            else:
                structured_review[section_key] = items
        
        return structured_review
    
    def _parse_issues(self, issue_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse issue lines into structured format.
        
        Args:
            issue_lines: Raw issue lines
            
        Returns:
            List of structured issues
        """
        issues = []
        
        for line in issue_lines:
            if not line:
                continue
            
            # Try to extract severity
            severity = "medium"  # default
            if any(word in line.lower() for word in ["critical", "high", "severe"]):
                severity = "high"
            elif any(word in line.lower() for word in ["low", "minor", "style"]):
                severity = "low"
            
            # Try to extract line number
            line_match = re.search(r"line\s+(\d+)", line, re.IGNORECASE)
            line_number = int(line_match.group(1)) if line_match else None
            
            issues.append({
                "description": line,
                "severity": severity,
                "line_number": line_number,
                "category": self._categorize_issue(line),
            })
        
        return issues
    
    def _categorize_issue(self, issue_description: str) -> str:
        """
        Categorize an issue based on its description.
        
        Args:
            issue_description: Issue description
            
        Returns:
            Issue category
        """
        description_lower = issue_description.lower()
        
        if any(word in description_lower for word in ["security", "vulnerability", "injection", "validation"]):
            return "security"
        elif any(word in description_lower for word in ["performance", "efficiency", "slow", "optimize"]):
            return "performance"
        elif any(word in description_lower for word in ["bug", "error", "wrong", "incorrect", "logic"]):
            return "correctness"
        elif any(word in description_lower for word in ["style", "format", "naming", "convention"]):
            return "style"
        elif any(word in description_lower for word in ["maintainability", "readable", "complex", "structure"]):
            return "maintainability"
        else:
            return "general"
    
    def _categorize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count issues by category.
        
        Args:
            issues: List of issues
            
        Returns:
            Dictionary with category counts
        """
        categories = {}
        for issue in issues:
            category = issue.get("category", "general")
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _filter_issues_by_severity(
        self,
        issues: List[Dict[str, Any]],
        severity_filter: str,
    ) -> List[Dict[str, Any]]:
        """
        Filter issues by severity level.
        
        Args:
            issues: List of issues
            severity_filter: Severity level to filter by
            
        Returns:
            Filtered list of issues
        """
        if severity_filter == "all":
            return issues
        
        return [issue for issue in issues if issue.get("severity") == severity_filter]
    
    def _detect_language(self, code: str) -> str:
        """
        Detect programming language from code.
        
        Args:
            code: Code to analyze
            
        Returns:
            Detected language
        """
        # Simple language detection based on syntax patterns
        if "def " in code and "import " in code:
            return "python"
        elif "function " in code or "const " in code or "=>" in code:
            return "javascript"
        elif "public class" in code or "private " in code:
            return "java"
        elif "#include" in code or "std::" in code:
            return "cpp"
        elif "func " in code and "package " in code:
            return "go"
        elif "fn " in code and "let " in code:
            return "rust"
        else:
            return "unknown"
    
    async def security_focused_review(
        self,
        code: str,
        **kwargs,
    ) -> AgentResponse:
        """
        Perform security-focused code review.
        
        Args:
            code: Code to review for security issues
            **kwargs: Additional parameters
            
        Returns:
            Security-focused review response
        """
        security_request = f"""**Security Code Review Request**

Please perform a comprehensive security review of the following code, focusing on:
- Input validation and sanitization
- Authentication and authorization issues
- Injection vulnerabilities (SQL, XSS, etc.)
- Cryptographic implementations
- Error handling and information disclosure
- Access control and privilege escalation
- Dependencies and third-party library usage

**Code:**
```
{code}
```

**Requirements:** Provide specific security recommendations with severity ratings."""
        
        return await self.process_request(
            security_request,
            context={"review_type": "security", "focus_areas": ["security"]},
            **kwargs,
        )
    
    async def performance_focused_review(
        self,
        code: str,
        **kwargs,
    ) -> AgentResponse:
        """
        Perform performance-focused code review.
        
        Args:
            code: Code to review for performance issues
            **kwargs: Additional parameters
            
        Returns:
            Performance-focused review response
        """
        performance_request = f"""**Performance Code Review Request**

Please perform a comprehensive performance review of the following code, focusing on:
- Algorithm efficiency and time complexity
- Memory usage and space complexity
- I/O operations optimization
- Database query performance
- Caching opportunities
- Concurrency and parallelization
- Resource cleanup and management

**Code:**
```
{code}
```

**Requirements:** Provide specific performance recommendations with impact estimates."""
        
        return await self.process_request(
            performance_request,
            context={"review_type": "performance", "focus_areas": ["performance"]},
            **kwargs,
        )