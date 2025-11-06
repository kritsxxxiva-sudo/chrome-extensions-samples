#!/usr/bin/env python3
"""
Awesome LLM Apps Repository Analyzer
Parses the directory structure and categorizes all projects
"""

import re
from collections import defaultdict
from pathlib import Path

def parse_structure(filename):
    """Parse the directory structure file"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Category definitions
    categories = {
        'single_agent': [],
        'multi_agent': [],
        'agent_teams': [],
        'mcp_agents': [],
        'rag_apps': [],
        'autonomous_gaming': [],
        'voice_apps': [],
        'framework_tutorials': [],
        'advanced_llm_apps': []
    }
    
    projects = []
    current_path = []
    
    for line in lines[1:]:  # Skip first line
        # Parse indentation and content
        match = re.match(r'(\s*)(├──|└──)\s+(.+)$', line)
        if not match:
            continue
            
        indent = len(match.group(1))
        name = match.group(3).strip()
        
        # Calculate depth based on indentation
        depth = indent // 4
        
        # Update current path
        if depth < len(current_path):
            current_path = current_path[:depth]
        
        if depth == len(current_path):
            if current_path:
                current_path[-1] = name
            else:
                current_path.append(name)
        else:
            current_path.append(name)
        
        # Track directories with README.md (indicating a project)
        if name == 'README.md' and len(current_path) >= 2:
            project_path = '/'.join(current_path[:-1])
            project_name = current_path[-2].replace('/', '')
            projects.append({
                'name': project_name,
                'path': project_path,
                'depth': depth
            })
    
    return projects

def categorize_projects(projects):
    """Categorize projects based on path patterns"""
    categories = defaultdict(list)
    
    for project in projects:
        path = project['path']
        name = project['name']
        
        if 'autonomous_game_playing' in path:
            categories['Autonomous Gaming Agents'].append(name)
        elif 'agent_teams' in path:
            categories['Multi-Agent Teams'].append(name)
        elif 'multi_agent' in path:
            categories['Multi-Agent Apps'].append(name)
        elif 'mcp_ai_agents' in path:
            categories['MCP Agents'].append(name)
        elif 'rag_tutorials' in path:
            categories['RAG Applications'].append(name)
        elif 'ai_agent_framework_crash_course' in path:
            categories['Framework Tutorials'].append(name)
        elif 'advanced_llm_apps' in path:
            categories['Advanced LLM Apps'].append(name)
        elif 'advanced_ai_agents' in path:
            categories['Advanced AI Agents'].append(name)
        else:
            categories['Other'].append(name)
    
    return categories

def generate_markdown(categories):
    """Generate comprehensive markdown documentation"""
    md = """# 🤖 Awesome LLM Apps - Complete Repository Analysis

> 🔍 Analyzing shubhamsaboo/awesome-llm-apps repository
> Complete catalog of AI agent implementations, frameworks, and tutorials

## 📋 Table of Contents
1. [Repository Overview](#repository-overview)
2. [Project Categories](#project-categories)
3. [Architecture Patterns](#architecture-patterns)
4. [Quick Reference](#quick-reference)
5. [Tech Stack Summary](#tech-stack-summary)

---

## 🎯 Repository Overview

The **awesome-llm-apps** repository is a comprehensive collection of LLM-powered applications showcasing various AI agent architectures, from simple single-agent systems to complex multi-agent orchestrations.

### Main Directories:
- **advanced_ai_agents/**: Advanced agent implementations (gaming, multi-agent systems)
- **advanced_llm_apps/**: Production-ready LLM applications
- **ai_agent_framework_crash_course/**: Educational framework tutorials
- **mcp_ai_agents/**: Model Context Protocol implementations
- **rag_tutorials/**: Retrieval Augmented Generation examples

---

## 📂 Project Categories

"""
    
    for category, items in sorted(categories.items()):
        if not items:
            continue
        md += f"\n### {category}\n\n"
        md += f"**Total Projects**: {len(items)}\n\n"
        for item in sorted(items):
            md += f"- `{item}`\n"
    
    md += """

---

## 🏗️ Architecture Patterns

### 1. **Single Agent Pattern**
Simple autonomous agents with tool usage capability.

```python
# Typical structure
agent = Agent(
    model="gpt-4",
    tools=[search_tool, calculator_tool],
    instructions="You are a helpful assistant"
)
response = agent.run("user query")
```

**Use Cases**: Simple automation, single-purpose tasks, quick prototypes

---

### 2. **Multi-Agent Team Pattern**
Multiple specialized agents collaborating on complex tasks.

```python
# Team coordination
team = Team(
    agents=[researcher, analyst, writer],
    workflow="sequential"  # or "parallel"
)
result = team.execute(task)
```

**Use Cases**: Complex workflows, domain expertise distribution, parallel processing

---

### 3. **MCP (Model Context Protocol) Pattern**
Standardized tool integration for agents.

```python
# MCP integration
mcp_server = MCPServer(tools=[filesystem, database])
agent = Agent(mcp_servers=[mcp_server])
```

**Use Cases**: Tool standardization, cross-agent tool sharing, enterprise integration

---

### 4. **RAG (Retrieval Augmented Generation) Pattern**
Knowledge-grounded generation with vector search.

```python
# RAG pipeline
vectordb = VectorDB(embeddings_model)
retriever = Retriever(vectordb)
agent = RAGAgent(retriever=retriever, llm=llm)
```

**Use Cases**: Knowledge-intensive tasks, document Q&A, semantic search

---

## 🔧 Tech Stack Summary

### Frameworks
- **PhiData**: Multi-agent orchestration
- **LangChain**: RAG and agent workflows
- **CrewAI**: Autonomous agent teams
- **AutoGen**: Conversational agents
- **AgentOps**: Agent monitoring & observability

### Models
- **OpenAI**: GPT-4, GPT-4-turbo, o1, o3
- **Anthropic**: Claude 3 Opus/Sonnet/Haiku
- **Open Source**: Llama 3, Mistral, Gemini

### Tools & Integrations
- **Vector DBs**: Pinecone, ChromaDB, FAISS, Qdrant
- **Search**: Tavily, DuckDuckGo, Google Search
- **Web**: Playwright, BeautifulSoup, Selenium
- **Data**: Pandas, Polars, SQL databases

---

## 🚀 Quick Reference

### Running a Simple Agent
```bash
cd single_agent_app
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python agent.py
```

### Running Multi-Agent Team
```bash
cd agent_teams/ai_finance_agent_team
pip install -r requirements.txt
python finance_agent_team.py
```

### Using MCP Agents
```bash
cd mcp_ai_agents
pip install -r requirements.txt
python mcp_agent.py
```

---

## 💡 Key Insights

1. **Agent Specialization**: Each agent should have a clear, focused role
2. **Tool Composition**: Combine simple tools for complex capabilities
3. **Memory Management**: Use appropriate memory strategies (short-term, long-term, semantic)
4. **Error Handling**: Implement retry logic and graceful degradation
5. **Observability**: Track agent decisions and tool usage

---

## 📊 Project Statistics

"""
    
    total = sum(len(items) for items in categories.values())
    md += f"- **Total Projects**: {total}\n"
    md += f"- **Categories**: {len([c for c, items in categories.items() if items])}\n"
    md += f"- **Frameworks Used**: PhiData, CrewAI, LangChain, AutoGen, AgentOps\n"
    md += f"- **Languages**: Python (primary), TypeScript (frontend)\n"
    
    return md

if __name__ == '__main__':
    print("Parsing repository structure...")
    projects = parse_structure('awesome-llm-apps-structure.txt')
    print(f"Found {len(projects)} projects")
    
    print("Categorizing projects...")
    categories = categorize_projects(projects)
    
    print("Generating documentation...")
    markdown = generate_markdown(categories)
    
    with open('AWESOME_LLM_APPS_ANALYSIS.md', 'w') as f:
        f.write(markdown)
    
    print("✅ Analysis complete! Check AWESOME_LLM_APPS_ANALYSIS.md")
    print(f"\nCategories found: {list(categories.keys())}")
