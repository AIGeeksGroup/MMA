# 文件: tests/test_confidence_v2.py (最终修正版)

import os
import time
from datetime import datetime
import pytest
import uuid # 导入uuid库来生成合法的ID

from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.agent import AgentState, AgentType, LLMConfig, Memory
from mirix.schemas.user import User as PydanticUser
from mirix.services.semantic_memory_manager import SemanticMemoryManager

from mirix.services.confidence_module import (
    reset_confidence_module,
    initialize_confidence_module,
    get_confidence_module,
)

def setup_module(module):
    """在当前测试文件所有测试开始前，运行一次，用于清理和初始化全局状态。"""
    print("\n--- Setting up test module ---")
    reset_confidence_module()
    # 确保配置文件路径正确
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "confidence_v2.yaml") # 假设是YAML
    if not os.path.exists(config_path):
        # 如果YAML不存在，尝试JSON作为备选
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "confidence_v2.json")
    
    initialize_confidence_module(config_path=config_path)

@pytest.fixture(scope="module")
def semantic_manager() -> SemanticMemoryManager:
    """创建一个可被所有测试用例复用的 SemanticMemoryManager 实例。"""
    return SemanticMemoryManager()

@pytest.fixture(scope="module")
def agent_state() -> AgentState:
    """创建一个可被所有测试用例复用的 AgentState 实例。"""
    emb = EmbeddingConfig.default_config(model_name="text-embedding-3-small")
    
    # --- START OF FIX for LLMConfig ---
    # 补上缺失的必填字段
    llm = LLMConfig(
        model="gpt-4o-mini", 
        endpoint_type="openai",
        model_endpoint_type="openai", # 补上 model_endpoint_type
        context_window=8192         # 补上 context_window
    )
    # --- END OF FIX ---

    return AgentState(
        id="agent_test", name="agent_conf_test", tool_rules=[], message_ids=[],
        system="", topic="", agent_type=AgentType.semantic_memory_agent,
        llm_config=llm, embedding_config=emb, organization_id="org_conf_test",
        description="", metadata_={}, memory=Memory(max_messages=128, blocks=[]),
        tools=[], tags=[], tool_exec_environment_variables=[]
    )

@pytest.fixture(scope="module")
def test_actor() -> PydanticUser:
    """创建一个用于测试的 Actor (User) 对象"""
    # --- START OF FIX for User ---
    # 生成一个符合格式的随机ID
    random_id = f"user-{uuid.uuid4().hex[:8]}"
    
    return PydanticUser(
        id=random_id, 
        organization_id="org_conf_test", 
        name="Test User",
        timezone="UTC"  # 添加缺失的 timezone 字段
    )
    # --- END OF FIX ---

# (test_v2_pipeline_functionality 函数保持不变)
def test_v2_pipeline_functionality(semantic_manager: SemanticMemoryManager, agent_state: AgentState, test_actor: PydanticUser):
    """
    端到端测试V2的创建、链接和置信度更新流程。
    """
    # ... (这部分代码是正确的，无需修改) ...
    from mirix.schemas.semantic_memory import SemanticMemoryItem
    
    cm = get_confidence_module()
    org_id = agent_state.organization_id

    item_data_1 = SemanticMemoryItem(name="API Docs", summary="The API returns JSON for user endpoints", details="User endpoints include create, update, delete", source="user", tree_path=["docs","api"], organization_id=org_id)
    item_data_2 = SemanticMemoryItem(name="API Documentation", summary="User endpoint responses are in JSON format", details="Contains create, update, delete methods", source="web", tree_path=["docs","api"], organization_id=org_id)
    item_data_3 = SemanticMemoryItem(name="Cooking Recipe", summary="This is a recipe for baking bread", details="Use flour, water, yeast, salt", source="user", tree_path=["hobby","cooking"], organization_id=org_id)
    
    created_items = semantic_manager.create_many_items(
        items=[item_data_1, item_data_2, item_data_3],
        actor=test_actor
    )
    
    a, b, c = created_items[0], created_items[1], created_items[2]

    assert a is not None and b is not None and c is not None, "Items must be retrievable"

    for it in [a, b, c]:
        assert it.confidence is not None, f"{it.id} confidence should not be None"
        ln = it.links
        assert ln and isinstance(ln, dict) and "neighbors" in ln, f"{it.id} links should have neighbors key"

    a_neighbor_ids = [n["target_id"] for n in a.links.get("neighbors", [])]
    assert b.id in a_neighbor_ids, "Item a should link to similar Item b"

    b_neighbor_ids = [n["target_id"] for n in b.links.get("neighbors", [])]
    assert a.id in b_neighbor_ids, "Item b should link to similar Item a"
    
    c_neighbor_ids = [n["target_id"] for n in c.links.get("neighbors", [])]
    assert a.id not in c_neighbor_ids, "Item c should not link to Item a"
    assert b.id not in c_neighbor_ids, "Item c should not link to Item b"

    v1_a = cm.compute_v1(a)
    v2_a = a.confidence
    print(f"\nItem 'API Docs' (ID: {a.id}):\n  - V1 Score: {v1_a:.4f}\n  - V2 Score: {v2_a:.4f}\n  - Links: {a_neighbor_ids}")

    v1_c = cm.compute_v1(c)
    v2_c = c.confidence
    print(f"Item 'Cooking Recipe' (ID: {c.id}):\n  - V1 Score: {v1_c:.4f}\n  - V2 Score: {v2_c:.4f}\n  - Links: {c_neighbor_ids}")
    
    assert v2_a > v1_a, "V2 score of a linked item should be pulled up by consensus"
    assert abs(v2_c - v1_c) < 0.1, "V2 score of an unlinked item should be close to its V1 score"

    print("\nAll assertions passed: V2 confidence pipeline appears functional.")