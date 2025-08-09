#!/usr/bin/env python3
"""Simple test script for HuggingFace model functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coding_framework.utils.config import load_config
from coding_framework.utils.llm_interface import LLMInterface
from langchain_core.messages import HumanMessage

async def test_model():
    """Test the HuggingFace model integration."""
    try:
        print("🔧 Loading configuration...")
        config = load_config()
        config.llm.provider = "huggingface"  
        config.llm.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        config.llm.temperature = 0.3
        
        print("🚀 Initializing HuggingFace model...")
        interface = LLMInterface(config.llm)
        await interface.initialize()
        print("✅ Model initialized successfully!")
        
        print("📝 Testing model generation...")
        response = await interface.call([HumanMessage(content="Generate a simple CUDA kernel for vector addition. Keep it concise.")])
        print(f"📋 Response ({len(response)} chars):")
        print(response[:500])
        if len(response) > 500:
            print("...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_model())
    sys.exit(0 if result else 1)