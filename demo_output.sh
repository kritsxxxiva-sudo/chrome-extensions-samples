#!/bin/bash
# Demo script to showcase the Expert AI Agent Architect System

echo "=================================="
echo "🤖 DEMO: Analyzing a Project"
echo "=================================="
echo ""
echo "Command: analyze ai_finance_agent_team"
echo ""
python3 -c "
from awesome_llm_agent_expert import AwesomeLLMExpert
expert = AwesomeLLMExpert()
expert.analyze_project('ai_finance_agent_team')
"

echo ""
echo "=================================="
echo "🤖 DEMO: Listing Categories"
echo "=================================="
echo ""
echo "Command: list"
echo ""
python3 -c "
from awesome_llm_agent_expert import AwesomeLLMExpert
expert = AwesomeLLMExpert()
expert.list_categories()
" | head -60

echo ""
echo "=================================="
echo "🤖 DEMO: Comparing Projects"
echo "=================================="
echo ""
echo "Command: compare finance legal"
echo ""
python3 -c "
from awesome_llm_agent_expert import AwesomeLLMExpert
expert = AwesomeLLMExpert()
expert.compare_projects('finance', 'legal')
"

echo ""
echo "=================================="
echo "🤖 DEMO: Generating New Use Case"
echo "=================================="
echo ""
echo "Command: generate healthcare"
echo ""
python3 -c "
from awesome_llm_agent_expert import AwesomeLLMExpert
expert = AwesomeLLMExpert()
expert.generate_new_use_case('healthcare')
" | head -80

echo ""
echo "=================================="
echo "✅ Demo Complete!"
echo "=================================="
echo ""
echo "Try it yourself:"
echo "  python3 awesome_llm_agent_expert.py"
echo ""
