import asyncio
from typing import Dict, Any, Optional, List
import structlog
import json

from ..agents.trainable_agent import TrainableAgent


class VERLRewardAdapter:
    """
    Adapter class to integrate our multi-agent reward system with VERL's training framework.
    
    This class bridges the gap between VERL's expected reward function interface
    and our multi-agent architecture, enabling distributed training while maintaining
    the benefits of specialized Generator, Reviewer, and Executor agents.
    """
    
    def __init__(
        self,
        generator_agent: TrainableAgent,
        reviewer_agent: Optional[TrainableAgent] = None,
        executor_agent: Optional[TrainableAgent] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        logger: Optional[Any] = None
    ):
        self.generator_agent = generator_agent
        self.reviewer_agent = reviewer_agent
        self.executor_agent = executor_agent
        
        self.reward_weights = reward_weights or {
            "correctness": 0.6,
            "code_quality": 0.2,
            "efficiency": 0.1,
            "review_score": 0.1
        }
        
        self.logger = logger or structlog.get_logger(component="verl_reward_adapter")
        
        # Validate reward weights
        if abs(sum(self.reward_weights.values()) - 1.0) > 0.01:
            self.logger.warning("Reward weights do not sum to 1.0, normalizing")
            total_weight = sum(self.reward_weights.values())
            self.reward_weights = {
                k: v / total_weight for k, v in self.reward_weights.items()
            }
            
        self.logger.info(
            "VERL reward adapter initialized",
            reward_weights=self.reward_weights,
            has_reviewer=reviewer_agent is not None,
            has_executor=executor_agent is not None
        )
        
    async def calculate_reward(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate reward for a prompt-response pair using multi-agent evaluation.
        
        This method is designed to be compatible with VERL's reward function interface
        while leveraging our multi-agent architecture for comprehensive evaluation.
        
        Args:
            prompt: The coding problem or prompt
            response: The generated code solution
            metadata: Additional context (test cases, expected output, etc.)
            
        Returns:
            Float reward score between 0.0 and 1.0
        """
        
        try:
            # Initialize reward components
            reward_components = {}
            
            # 1. Correctness evaluation using executor agent
            if self.executor_agent and metadata and metadata.get("test_cases"):
                correctness_reward = await self._evaluate_correctness(
                    prompt, response, metadata.get("test_cases", [])
                )
                reward_components["correctness"] = correctness_reward
            else:
                # Fallback to basic syntax check
                reward_components["correctness"] = self._basic_correctness_check(response)
                
            # 2. Code quality evaluation using reviewer agent
            if self.reviewer_agent:
                quality_reward = await self._evaluate_code_quality(prompt, response)
                reward_components["code_quality"] = quality_reward
            else:
                # Fallback to basic quality metrics
                reward_components["code_quality"] = self._basic_quality_check(response)
                
            # 3. Efficiency evaluation (basic implementation)
            reward_components["efficiency"] = self._evaluate_efficiency(response)
            
            # 4. Review score (if reviewer agent available)
            if self.reviewer_agent:
                review_reward = await self._get_review_score(prompt, response)
                reward_components["review_score"] = review_reward
            else:
                reward_components["review_score"] = reward_components["code_quality"]
                
            # Calculate weighted total reward
            total_reward = sum(
                self.reward_weights.get(component, 0.0) * score
                for component, score in reward_components.items()
            )
            
            # Ensure reward is in [0, 1] range
            total_reward = max(0.0, min(1.0, total_reward))
            
            self.logger.debug(
                "Reward calculated",
                prompt_length=len(prompt),
                response_length=len(response),
                reward_components=reward_components,
                total_reward=total_reward
            )
            
            return total_reward
            
        except Exception as e:
            self.logger.error(f"Reward calculation failed: {e}")
            return 0.0  # Return 0 reward on failure
            
    async def calculate_multi_turn_reward(
        self,
        conversation_history: List[Dict[str, str]],
        final_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate reward for multi-turn conversations.
        
        This method handles the reward calculation for conversations involving
        multiple turns between agents (e.g., Generator -> Reviewer -> Generator).
        
        Args:
            conversation_history: List of conversation turns
            final_response: The final code solution
            metadata: Additional context and test cases
            
        Returns:
            Float reward score between 0.0 and 1.0
        """
        
        try:
            # Extract the original prompt
            original_prompt = ""
            if conversation_history:
                original_prompt = conversation_history[0].get("content", "")
                
            # Calculate reward for final response
            final_reward = await self.calculate_reward(
                original_prompt, final_response, metadata
            )
            
            # Apply multi-turn bonus/penalty based on conversation quality
            conversation_bonus = self._evaluate_conversation_quality(conversation_history)
            
            # Combine final reward with conversation quality
            total_reward = (final_reward * 0.8) + (conversation_bonus * 0.2)
            
            self.logger.debug(
                "Multi-turn reward calculated",
                conversation_turns=len(conversation_history),
                final_reward=final_reward,
                conversation_bonus=conversation_bonus,
                total_reward=total_reward
            )
            
            return max(0.0, min(1.0, total_reward))
            
        except Exception as e:
            self.logger.error(f"Multi-turn reward calculation failed: {e}")
            return 0.0
            
    async def _evaluate_correctness(
        self, 
        prompt: str, 
        response: str, 
        test_cases: List[Dict[str, Any]]
    ) -> float:
        """Evaluate code correctness using executor agent."""
        
        if not self.executor_agent:
            return self._basic_correctness_check(response)
            
        try:
            # Use executor agent to run test cases
            execution_context = {
                "code": response,
                "test_cases": test_cases,
                "timeout": 10,  # 10 second timeout
            }
            
            execution_result = await self.executor_agent.process_request(
                f"Execute and test this code:\n{response}",
                context=execution_context
            )
            
            if execution_result.success:
                # Parse execution results
                results = execution_result.metadata or {}
                passed_tests = results.get("passed_tests", 0)
                total_tests = results.get("total_tests", len(test_cases))
                
                if total_tests > 0:
                    return passed_tests / total_tests
                else:
                    return 1.0 if execution_result.success else 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Correctness evaluation failed: {e}")
            return self._basic_correctness_check(response)
            
    async def _evaluate_code_quality(self, prompt: str, response: str) -> float:
        """Evaluate code quality using reviewer agent."""
        
        if not self.reviewer_agent:
            return self._basic_quality_check(response)
            
        try:
            review_request = f"""
Please review this code solution:

Problem: {prompt}

Solution:
{response}

Provide a quality score between 0.0 and 1.0.
"""
            
            review_result = await self.reviewer_agent.process_request(
                review_request,
                context={
                    "review_mode": True,
                    "return_score": True,
                }
            )
            
            if review_result.success:
                # Try to extract numeric score from review
                score = self._extract_score_from_review(review_result.content)
                return score if score is not None else 0.5
            else:
                return self._basic_quality_check(response)
                
        except Exception as e:
            self.logger.warning(f"Code quality evaluation failed: {e}")
            return self._basic_quality_check(response)
            
    def _basic_correctness_check(self, code: str) -> float:
        """Basic syntax and structure check."""
        
        if not code or not code.strip():
            return 0.0
            
        try:
            # Try to parse as Python code
            import ast
            ast.parse(code)
            
            # Basic checks
            score = 0.0
            
            # Has function definition
            if "def " in code:
                score += 0.3
                
            # Has return statement
            if "return " in code:
                score += 0.2
                
            # Has some logic (if, for, while, etc.)
            if any(keyword in code for keyword in ["if ", "for ", "while ", "try "]):
                score += 0.2
                
            # Not too short or too long
            lines = code.strip().split('\n')
            if 3 <= len(lines) <= 50:
                score += 0.3
            elif len(lines) > 0:
                score += 0.1
                
            return min(1.0, score)
            
        except SyntaxError:
            return 0.0
        except Exception:
            return 0.1  # Give small reward for any attempt
            
    def _basic_quality_check(self, code: str) -> float:
        """Basic code quality assessment."""
        
        if not code or not code.strip():
            return 0.0
            
        score = 0.0
        lines = code.strip().split('\n')
        
        # Check for comments
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        if comment_lines > 0:
            score += 0.2
            
        # Check for docstrings
        if '"""' in code or "'''" in code:
            score += 0.2
            
        # Check for appropriate variable names (not single characters)
        import re
        var_names = re.findall(r'\b([a-z_]\w*)\s*=', code.lower())
        meaningful_vars = [name for name in var_names if len(name) > 2 and name != 'tmp']
        if meaningful_vars:
            score += 0.2
            
        # Check line length (reasonable formatting)
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines / max(len(lines), 1) < 0.2:  # Less than 20% long lines
            score += 0.2
            
        # Check for error handling
        if "try:" in code or "except" in code:
            score += 0.2
            
        return min(1.0, score)
        
    def _evaluate_efficiency(self, code: str) -> float:
        """Basic efficiency evaluation."""
        
        if not code:
            return 0.0
            
        # Simple heuristics for efficiency
        score = 0.5  # Start with neutral score
        
        # Penalize nested loops
        nested_loop_count = code.count("for ") * code.count("while ")
        if nested_loop_count > 2:
            score -= 0.2
            
        # Reward list comprehensions
        if "[" in code and "for " in code and "]" in code:
            score += 0.2
            
        # Penalize excessive function calls in loops
        if "for " in code and code.count("(") > 5:
            score -= 0.1
            
        return max(0.0, min(1.0, score))
        
    async def _get_review_score(self, prompt: str, response: str) -> float:
        """Get review score from reviewer agent."""
        
        # This would typically be the same as code quality evaluation
        # but could be customized for different review criteria
        return await self._evaluate_code_quality(prompt, response)
        
    def _extract_score_from_review(self, review_text: str) -> Optional[float]:
        """Extract numeric score from review text."""
        
        import re
        
        # Look for patterns like "score: 0.8", "rating: 0.7", "8/10", "0.85"
        patterns = [
            r'score[:\s]+(\d+\.?\d*)',
            r'rating[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)/10',
            r'(\d+\.?\d*)\s*out\s*of\s*10',
            r'\b(\d*\.\d+)\b',  # Any decimal number
            r'\b([0-9])\s*/\s*10',  # Integer out of 10
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, review_text.lower())
            if matches:
                try:
                    score = float(matches[0])
                    # Normalize to 0-1 range
                    if score > 1.0:
                        score = score / 10.0  # Assume it was out of 10
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue
                    
        return None
        
    def _evaluate_conversation_quality(
        self, 
        conversation_history: List[Dict[str, str]]
    ) -> float:
        """Evaluate the quality of multi-turn conversation."""
        
        if not conversation_history:
            return 0.5
            
        # Simple heuristics for conversation quality
        score = 0.5
        
        # Reward longer conversations (up to a point)
        turns = len(conversation_history)
        if 2 <= turns <= 5:
            score += 0.2
        elif turns > 5:
            score -= 0.1  # Too many turns might indicate confusion
            
        # Check for improvement over turns (basic heuristic)
        if turns >= 2:
            first_response = conversation_history[0].get("content", "")
            last_response = conversation_history[-1].get("content", "")
            
            if len(last_response) > len(first_response):
                score += 0.1  # Longer final response might indicate improvement
                
        # Check for reviewer feedback incorporation
        has_reviewer_input = any(
            "review" in turn.get("role", "").lower() or 
            "feedback" in turn.get("content", "").lower()
            for turn in conversation_history
        )
        
        if has_reviewer_input:
            score += 0.2
            
        return max(0.0, min(1.0, score))