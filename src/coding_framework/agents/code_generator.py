"""
Code Generator Agent for creating initial code solutions.

This agent specializes in generating code solutions from problem descriptions,
supporting multiple programming languages and coding patterns.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from .base_agent import BaseAgent, AgentResponse


class CodeGeneratorAgent(BaseAgent):
    """
    Specialized agent for code generation.
    
    Generates initial code solutions from problem descriptions, supports
    multiple programming languages, and can adapt coding style based on context.
    """
    
    @property
    def agent_type(self) -> str:
        """Return agent type identifier."""
        return "code_generator"
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for code generation."""
        return """You are a Senior Software Engineer specializing in code generation. Your role is to:

1. **Understand Requirements**: Carefully analyze problem descriptions and requirements
2. **Generate Clean Code**: Write well-structured, readable, and efficient code
3. **Follow Best Practices**: Use appropriate design patterns and coding conventions
4. **Provide Explanations**: Include comments and brief explanations for complex logic
5. **Consider Edge Cases**: Think about potential edge cases and handle them appropriately

**Key Guidelines:**
- Write production-ready code, not just working code
- Use meaningful variable and function names
- Include appropriate error handling
- Follow language-specific conventions and best practices
- Optimize for readability and maintainability
- Include docstrings/comments for functions and classes

**Output Format:**
Always structure your response as:
```language
# Brief explanation of the approach
[Your code here]
```

**Languages Supported:** Python, JavaScript, TypeScript, Java, C++, Go, Rust
**Default Language:** Python (unless specified otherwise)"""
    
    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Generate code solution from problem description.
        
        Args:
            request: Problem description or requirements
            context: Additional context (language, constraints, etc.)
            **kwargs: Additional generation parameters
            
        Returns:
            Agent response with generated code
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting code generation", request_length=len(request))
            
            # Extract generation parameters
            language = kwargs.get("language") or (context or {}).get("language", "python")
            style = kwargs.get("style") or (context or {}).get("style", "clean")
            include_tests = kwargs.get("include_tests", False)
            
            # Update state with current request
            self.update_state("last_request", request)
            self.update_state("target_language", language)
            
            # Build enhanced prompt
            enhanced_request = self._build_enhanced_request(
                request, language, style, include_tests
            )
            
            # Prepare messages for LLM
            messages = self._build_messages(enhanced_request, context)
            
            # Generate code using LLM
            response = await self._call_llm(
                messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            
            # Process and validate the generated code
            code_info = self._extract_code_from_response(response)
            
            # Create response metadata
            metadata = {
                "language": language,
                "style": style,
                "include_tests": include_tests,
                "code_length": len(code_info["code"]),
                "has_explanation": bool(code_info["explanation"]),
                "detected_functions": self._detect_functions(code_info["code"], language),
                "estimated_complexity": self._estimate_complexity(code_info["code"]),
            }
            
            # Add to conversation history
            self.add_to_history(HumanMessage(content=request))
            self.add_to_history(HumanMessage(content=response))
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                "Code generation completed",
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
                "Code generation failed",
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
    
    def _build_enhanced_request(
        self,
        request: str,
        language: str,
        style: str,
        include_tests: bool,
    ) -> str:
        """
        Build enhanced request with specific requirements.
        
        Args:
            request: Original problem description
            language: Target programming language
            style: Coding style preference
            include_tests: Whether to include test cases
            
        Returns:
            Enhanced request string
        """
        enhanced_parts = [f"**Problem:** {request}"]
        
        enhanced_parts.append(f"**Target Language:** {language.title()}")
        
        if style != "clean":
            enhanced_parts.append(f"**Style Preference:** {style}")
        
        if include_tests:
            enhanced_parts.append("**Requirements:** Include unit tests for the solution")
        
        # Add language-specific requirements
        lang_requirements = self._get_language_requirements(language)
        if lang_requirements:
            enhanced_parts.append(f"**Language-Specific Requirements:** {lang_requirements}")
        
        return "\n\n".join(enhanced_parts)
    
    def _get_language_requirements(self, language: str) -> str:
        """
        Get language-specific coding requirements.
        
        Args:
            language: Programming language
            
        Returns:
            Language-specific requirements string
        """
        requirements = {
            "python": "Follow PEP 8, use type hints, include docstrings",
            "javascript": "Use modern ES6+ syntax, proper error handling",
            "typescript": "Use strict typing, interfaces for complex objects",
            "java": "Follow Oracle conventions, use proper access modifiers",
            "cpp": "Use modern C++17/20 features, RAII principles",
            "go": "Follow Go conventions, proper error handling with errors package",
            "rust": "Use ownership principles, proper error handling with Result type",
        }
        
        return requirements.get(language.lower(), "Follow language best practices")
    
    def _extract_code_from_response(self, response: str) -> Dict[str, str]:
        """
        Extract code and explanation from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Dictionary with code and explanation
        """
        # Look for code blocks
        code_block_pattern = r"```(?:\w+\n)?(.*?)```"
        code_matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if code_matches:
            # Take the largest code block (most likely the main solution)
            code = max(code_matches, key=len).strip()
            
            # Extract explanation (text before the first code block)
            explanation_match = re.split(r"```", response, 1)
            explanation = explanation_match[0].strip() if explanation_match else ""
        else:
            # Fallback: treat entire response as code if no code blocks found
            code = response.strip()
            explanation = ""
        
        return {
            "code": code,
            "explanation": explanation,
        }
    
    def _detect_functions(self, code: str, language: str) -> List[str]:
        """
        Detect function definitions in the generated code.
        
        Args:
            code: Generated code
            language: Programming language
            
        Returns:
            List of detected function names
        """
        function_patterns = {
            "python": r"def\s+(\w+)",
            "javascript": r"function\s+(\w+)|const\s+(\w+)\s*=|(\w+)\s*:\s*function",
            "typescript": r"function\s+(\w+)|const\s+(\w+)\s*=|(\w+)\s*:\s*\(",
            "java": r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(",
            "cpp": r"(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*{",
            "go": r"func\s+(\w+)",
            "rust": r"fn\s+(\w+)",
        }
        
        pattern = function_patterns.get(language.lower(), r"\w+\s+(\w+)\s*\(")
        matches = re.findall(pattern, code, re.MULTILINE)
        
        # Flatten nested matches and filter out empty strings
        functions = []
        for match in matches:
            if isinstance(match, tuple):
                functions.extend([m for m in match if m])
            else:
                functions.append(match)
        
        return list(set(functions))  # Remove duplicates
    
    def _estimate_complexity(self, code: str) -> str:
        """
        Provide a rough estimate of code complexity.
        
        Args:
            code: Generated code
            
        Returns:
            Complexity estimate (simple, moderate, complex)
        """
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        line_count = len(lines)
        
        # Count complexity indicators
        complexity_indicators = [
            "for ", "while ", "if ", "elif ", "else:",
            "try:", "except:", "class ", "def ",
            "lambda", "comprehension", "recursive",
        ]
        
        complexity_score = sum(
            code.lower().count(indicator) for indicator in complexity_indicators
        )
        
        if line_count < 20 and complexity_score < 5:
            return "simple"
        elif line_count < 50 and complexity_score < 15:
            return "moderate"
        else:
            return "complex"
    
    async def generate_variations(
        self,
        request: str,
        num_variations: int = 3,
        **kwargs,
    ) -> List[AgentResponse]:
        """
        Generate multiple variations of a solution.
        
        Args:
            request: Problem description
            num_variations: Number of variations to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of agent responses with different solutions
        """
        variations = []
        
        for i in range(num_variations):
            # Vary the temperature for different approaches
            temp_kwargs = kwargs.copy()
            temp_kwargs["temperature"] = 0.3 + (i * 0.3)  # 0.3, 0.6, 0.9
            
            # Add variation instruction
            variation_request = f"{request}\n\n**Note:** Provide a different approach or implementation style for this solution (variation {i+1})."
            
            response = await self.process_request(variation_request, **temp_kwargs)
            variations.append(response)
        
        return variations
    
    async def optimize_code(
        self,
        code: str,
        optimization_focus: str = "performance",
        **kwargs,
    ) -> AgentResponse:
        """
        Optimize existing code for specific criteria.
        
        Args:
            code: Code to optimize
            optimization_focus: Focus area (performance, readability, memory)
            **kwargs: Additional parameters
            
        Returns:
            Agent response with optimized code
        """
        optimization_request = f"""**Task:** Optimize the following code focusing on {optimization_focus}.

**Original Code:**
```
{code}
```

**Requirements:**
- Maintain the same functionality
- Focus on {optimization_focus} improvements
- Explain the optimizations made
- Provide before/after comparison if significant changes are made"""
        
        return await self.process_request(
            optimization_request,
            context={"task": "optimization", "focus": optimization_focus},
            **kwargs,
        )