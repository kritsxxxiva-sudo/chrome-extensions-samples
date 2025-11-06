#!/usr/bin/env python3
"""
🤖 Awesome LLM Apps - Expert AI Agent Architect

Your AI assistant for understanding and working with the shubhamsaboo/awesome-llm-apps repository.
Deep knowledge of agent patterns, frameworks, and architectures.
"""

import re
import json
from pathlib import Path

class AwesomeLLMExpert:
    """Expert AI Agent Architect for awesome-llm-apps repository"""
    
    def __init__(self):
        self.projects = self._load_projects()
        
    def _load_projects(self):
        """Load and parse all projects from the structure file"""
        projects = []
        
        try:
            with open('awesome-llm-apps-structure.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract all project directories (those with README.md)
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if 'README.md' in line and i > 0:
                    # Look back to find the directory name
                    for j in range(i-1, max(0, i-5), -1):
                        if '/' in lines[j] and 'README.md' not in lines[j]:
                            # Extract project name
                            name_match = re.search(r'([a-z_0-9]+)/', lines[j])
                            if name_match:
                                project_name = name_match.group(1)
                                
                                # Determine category from path context
                                context = '\n'.join(lines[max(0, j-10):j])
                                category = self._determine_category(project_name, context)
                                
                                projects.append({
                                    'name': project_name,
                                    'category': category
                                })
                            break
            
            # Remove duplicates
            unique_projects = []
            seen = set()
            for p in projects:
                if p['name'] not in seen:
                    seen.add(p['name'])
                    unique_projects.append(p)
            
            return unique_projects
            
        except FileNotFoundError:
            print("⚠️  Structure file not found")
            return []
    
    def _determine_category(self, name, context):
        """Determine project category"""
        context_lower = context.lower() + name.lower()
        
        if 'game' in name or 'chess' in name or 'tic_tac_toe' in name or '3dpygame' in name:
            return 'Autonomous Gaming'
        elif 'team' in name or 'agent_teams' in context_lower:
            return 'Multi-Agent Teams'
        elif 'mcp' in name or 'mcp_' in context_lower:
            return 'MCP Agents'
        elif 'rag' in name or 'rag_' in context_lower:
            return 'RAG Applications'
        elif 'multimodal' in name or 'vision' in name:
            return 'Multimodal Agents'
        elif 'voice' in name or 'audio' in name or 'speech' in name:
            return 'Voice Agents'
        elif 'framework' in context_lower or 'crash_course' in context_lower:
            return 'Framework Tutorials'
        elif 'multi_agent' in context_lower:
            return 'Multi-Agent Systems'
        else:
            return 'Single Agent'
    
    def analyze_project(self, project_name):
        """Analyze a specific project"""
        print(f"\n> 🔍 Analyzing Awesome LLM App: **{project_name}**\n")
        
        project = next((p for p in self.projects if project_name.lower() in p['name'].lower()), None)
        
        if not project:
            print(f"❌ Project '{project_name}' not found\n")
            print("💡 Try one of these:")
            for p in self.projects[:10]:
                print(f"   - {p['name']}")
            return
        
        # Generate detailed analysis
        self._print_project_details(project)
    
    def _print_project_details(self, project):
        """Print comprehensive project details"""
        name = project['name']
        category = project['category']
        
        print(f"## 📋 Project: `{name}`\n")
        print(f"**Category**: {category}\n")
        
        # Provide architecture based on category
        if category == 'Multi-Agent Teams':
            self._explain_multi_agent_team(name)
        elif category == 'RAG Applications':
            self._explain_rag_app(name)
        elif category == 'MCP Agents':
            self._explain_mcp_agent(name)
        elif category == 'Autonomous Gaming':
            self._explain_gaming_agent(name)
        elif category == 'Voice Agents':
            self._explain_voice_agent(name)
        else:
            self._explain_single_agent(name)
    
    def _explain_multi_agent_team(self, name):
        """Explain multi-agent team architecture"""
        print("### 🏗️ Architecture: Multi-Agent Team\n")
        print("```")
        print("User Input")
        print("    ↓")
        print("Agent 1 (Specialist) → Agent 2 (Analyst) → Agent 3 (Synthesizer)")
        print("    ↓                      ↓                      ↓")
        print("Task Execution  →  Analysis & Processing  →  Final Output")
        print("```\n")
        
        print("### 🔧 Tech Stack\n")
        print("- **Framework**: PhiData / CrewAI")
        print("- **Model**: GPT-4 / Claude 3")
        print("- **Pattern**: Sequential or Parallel Workflow")
        print("- **Tools**: DuckDuckGo, Python, Custom APIs\n")
        
        print("### 💡 Key Concepts\n")
        print("- **Agent Specialization**: Each agent has a focused role")
        print("- **Workflow Coordination**: Sequential or parallel execution")
        print("- **Shared Memory**: Agents can access previous outputs")
        print("- **Tool Sharing**: Common tools across agents\n")
        
        print("### 📝 Example Usage\n")
        print("```python")
        print("from phi.agent import Agent, Team")
        print("")
        print("# Define specialized agents")
        print("researcher = Agent(name='Researcher', role='Gather information')")
        print("analyst = Agent(name='Analyst', role='Analyze data')")
        print("writer = Agent(name='Writer', role='Create report')")
        print("")
        print("# Create team")
        print("team = Team(agents=[researcher, analyst, writer])")
        print("result = team.run('Analyze market trends')")
        print("```\n")
    
    def _explain_rag_app(self, name):
        """Explain RAG application architecture"""
        print("### 🏗️ Architecture: RAG (Retrieval Augmented Generation)\n")
        print("```")
        print("Documents → Chunking → Embeddings → Vector DB")
        print("                                        ↓")
        print("User Query → Embedding → Similarity Search")
        print("                                        ↓")
        print("Retrieved Chunks → Context → LLM → Answer")
        print("```\n")
        
        print("### 🔧 Tech Stack\n")
        print("- **Framework**: LangChain / LlamaIndex / PhiData")
        print("- **Vector DB**: ChromaDB / Pinecone / FAISS")
        print("- **Embeddings**: OpenAI / Cohere / Sentence Transformers")
        print("- **LLM**: GPT-4 / Claude 3\n")
        
        print("### 💡 Key Concepts\n")
        print("- **Chunking Strategy**: Optimal document splitting")
        print("- **Embedding Model**: Convert text to vectors")
        print("- **Similarity Search**: Find relevant context")
        print("- **Context Injection**: Provide grounding to LLM\n")
        
        print("### 📝 Example Usage\n")
        print("```python")
        print("from phi.knowledge.pdf import PDFKnowledgeBase")
        print("from phi.vectordb.chroma import ChromaDB")
        print("from phi.agent import Agent")
        print("")
        print("# Create knowledge base")
        print("kb = PDFKnowledgeBase(")
        print("    path='documents/',")
        print("    vector_db=ChromaDB(collection='docs')")
        print(")")
        print("kb.load()")
        print("")
        print("# Create RAG agent")
        print("agent = Agent(knowledge_base=kb, search_knowledge=True)")
        print("answer = agent.run('What does the document say about X?')")
        print("```\n")
    
    def _explain_mcp_agent(self, name):
        """Explain MCP agent architecture"""
        print("### 🏗️ Architecture: MCP (Model Context Protocol)\n")
        print("```")
        print("Agent → MCP Protocol → Tool Registry")
        print("                           ↓")
        print("        ┌─────────────┬────────────┬─────────────┐")
        print("        ↓             ↓            ↓             ↓")
        print("   Filesystem     Database      API        Custom Tools")
        print("```\n")
        
        print("### 🔧 Tech Stack\n")
        print("- **Protocol**: Model Context Protocol (MCP)")
        print("- **Framework**: MCP SDK")
        print("- **Tools**: Standardized tool interfaces")
        print("- **Integration**: Cross-platform compatibility\n")
        
        print("### 💡 Key Concepts\n")
        print("- **Standardization**: Unified tool interface")
        print("- **Interoperability**: Tools work across different agents")
        print("- **Discovery**: Automatic tool registration")
        print("- **Composition**: Combine tools flexibly\n")
    
    def _explain_gaming_agent(self, name):
        """Explain gaming agent architecture"""
        print("### 🏗️ Architecture: Autonomous Gaming Agent\n")
        print("```")
        print("Game State → Agent Perception")
        print("                  ↓")
        print("          Strategy Analysis")
        print("                  ↓")
        print("          Move Decision → Game Action")
        print("                  ↓")
        print("            Update State")
        print("                  ↓")
        print("            [Loop continues]")
        print("```\n")
        
        print("### 🔧 Tech Stack\n")
        print("- **AI Model**: GPT-4 / Claude (strategic thinking)")
        print("- **Game Engine**: Python-based game logic")
        print("- **Tools**: Move validator, board analyzer")
        print("- **Memory**: Game state tracking\n")
        
        print("### 💡 Key Concepts\n")
        print("- **State Representation**: Encode game state for LLM")
        print("- **Strategic Reasoning**: LLM analyzes optimal moves")
        print("- **Action Execution**: Validate and execute moves")
        print("- **Learning Loop**: Improve through gameplay\n")
    
    def _explain_voice_agent(self, name):
        """Explain voice agent architecture"""
        print("### 🏗️ Architecture: Voice Agent\n")
        print("```")
        print("Audio Input → Speech-to-Text → LLM Processing")
        print("                                     ↓")
        print("               Text-to-Speech ← Response Generation")
        print("                     ↓")
        print("               Audio Output")
        print("```\n")
        
        print("### 🔧 Tech Stack\n")
        print("- **STT**: OpenAI Whisper / AssemblyAI")
        print("- **LLM**: GPT-4 / Claude")
        print("- **TTS**: ElevenLabs / OpenAI TTS")
        print("- **Audio Processing**: PyAudio / SoundDevice\n")
    
    def _explain_single_agent(self, name):
        """Explain single agent architecture"""
        print("### 🏗️ Architecture: Single Agent with Tools\n")
        print("```")
        print("User Query → Agent (LLM)")
        print("                ↓")
        print("        Tool Selection")
        print("                ↓")
        print("         Tool Execution")
        print("                ↓")
        print("      Response Generation")
        print("```\n")
        
        print("### 🔧 Tech Stack\n")
        print("- **Framework**: PhiData / LangChain")
        print("- **Model**: GPT-4 / Claude 3")
        print("- **Tools**: Search, Calculator, APIs")
        print("- **Memory**: Conversation history\n")
    
    def list_categories(self):
        """List all project categories"""
        print("\n## 📂 Project Categories\n")
        
        categories = {}
        for project in self.projects:
            cat = project['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(project['name'])
        
        for cat in sorted(categories.keys()):
            print(f"\n### {cat} ({len(categories[cat])} projects)\n")
            for name in sorted(categories[cat])[:10]:  # Show first 10
                print(f"- `{name}`")
            if len(categories[cat]) > 10:
                print(f"- ... and {len(categories[cat]) - 10} more")
    
    def compare_projects(self, name1, name2):
        """Compare two projects"""
        print(f"\n> 🔍 Comparing: **{name1}** vs **{name2}**\n")
        
        p1 = next((p for p in self.projects if name1.lower() in p['name'].lower()), None)
        p2 = next((p for p in self.projects if name2.lower() in p['name'].lower()), None)
        
        if not p1 or not p2:
            print("❌ One or both projects not found\n")
            return
        
        print("| Aspect | " + p1['name'] + " | " + p2['name'] + " |")
        print("|--------|" + "-" * len(p1['name']) + "|" + "-" * len(p2['name']) + "|")
        print(f"| Category | {p1['category']} | {p2['category']} |")
        print(f"| Complexity | {'High' if 'team' in p1['name'] else 'Medium'} | {'High' if 'team' in p2['name'] else 'Medium'} |")
        print(f"| Use Case | {self._guess_use_case(p1['name'])} | {self._guess_use_case(p2['name'])} |")
        print()
    
    def _guess_use_case(self, name):
        """Guess use case from project name"""
        if 'finance' in name:
            return 'Financial Analysis'
        elif 'legal' in name:
            return 'Legal Research'
        elif 'travel' in name:
            return 'Travel Planning'
        elif 'recruitment' in name:
            return 'HR & Recruiting'
        elif 'game' in name:
            return 'Gaming & Entertainment'
        elif 'research' in name:
            return 'Research & Analysis'
        else:
            return 'General Purpose'
    
    def generate_new_use_case(self, domain):
        """Generate new use case inspired by existing patterns"""
        print(f"\n> 💡 Generating New Use Case: **AI {domain.title()} Agent**\n")
        
        print(f"## 🎯 Concept: AI {domain.title()} Agent Team\n")
        print(f"**Inspired by**: Multi-agent team patterns in the repository\n")
        
        print("### 🏗️ Architecture\n")
        print("```")
        print(f"{domain.title()} Request")
        print("    ↓")
        print(f"Research Agent → Analysis Agent → Action Agent → Report Agent")
        print("```\n")
        
        print("### 👥 Agent Roles\n")
        print(f"1. **Research Agent**: Gather {domain}-related information")
        print(f"2. **Analysis Agent**: Analyze {domain} data and patterns")
        print(f"3. **Action Agent**: Generate actionable recommendations")
        print(f"4. **Report Agent**: Create comprehensive {domain} report\n")
        
        print("### 📝 Code Example\n")
        print("```python")
        print("from phi.agent import Agent, Team")
        print("from phi.tools import DuckDuckGo, PythonTools")
        print("")
        print("# Define agents")
        print(f"research_agent = Agent(")
        print(f"    name='{domain.title()} Researcher',")
        print(f"    role='Research {domain} information',")
        print("    tools=[DuckDuckGo()]")
        print(")")
        print("")
        print(f"analysis_agent = Agent(")
        print(f"    name='{domain.title()} Analyst',")
        print(f"    role='Analyze {domain} data',")
        print("    tools=[PythonTools()]")
        print(")")
        print("")
        print(f"action_agent = Agent(")
        print(f"    name='{domain.title()} Strategist',")
        print(f"    role='Generate {domain} recommendations'")
        print(")")
        print("")
        print(f"report_agent = Agent(")
        print(f"    name='{domain.title()} Reporter',")
        print(f"    role='Create {domain} report'")
        print(")")
        print("")
        print("# Create team")
        print(f"{domain}_team = Team(")
        print("    agents=[research_agent, analysis_agent, action_agent, report_agent],")
        print("    workflow='sequential'")
        print(")")
        print("")
        print(f"# Execute")
        print(f"result = {domain}_team.run('Analyze {domain} trends')")
        print("```\n")
        
        print("### 🎨 Architecture Diagram\n")
        print("```")
        print("┌─────────────────────────────────────────────┐")
        print("│          User Input / Requirements          │")
        print("└─────────────────┬───────────────────────────┘")
        print("                  ↓")
        print("┌─────────────────────────────────────────────┐")
        print(f"│         Research Agent ({domain})            │")
        print("│  • Web search                               │")
        print("│  • Data collection                          │")
        print("└─────────────────┬───────────────────────────┘")
        print("                  ↓")
        print("┌─────────────────────────────────────────────┐")
        print(f"│         Analysis Agent ({domain})            │")
        print("│  • Data processing                          │")
        print("│  • Pattern recognition                      │")
        print("└─────────────────┬───────────────────────────┘")
        print("                  ↓")
        print("┌─────────────────────────────────────────────┐")
        print(f"│          Action Agent ({domain})             │")
        print("│  • Strategy generation                      │")
        print("│  • Recommendations                          │")
        print("└─────────────────┬───────────────────────────┘")
        print("                  ↓")
        print("┌─────────────────────────────────────────────┐")
        print(f"│          Report Agent ({domain})             │")
        print("│  • Report generation                        │")
        print("│  • Visualization                            │")
        print("└─────────────────┬───────────────────────────┘")
        print("                  ↓")
        print("┌─────────────────────────────────────────────┐")
        print(f"│            Final {domain.title()} Report        │")
        print("└─────────────────────────────────────────────┘")
        print("```\n")
    
    def show_stats(self):
        """Show repository statistics"""
        print("\n## 📊 Repository Statistics\n")
        print(f"- **Total Projects**: {len(self.projects)}")
        
        categories = {}
        for p in self.projects:
            cat = p['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"- **Categories**: {len(categories)}")
        print(f"- **Frameworks**: PhiData, CrewAI, LangChain, AutoGen")
        print(f"- **Languages**: Python (primary), TypeScript (frontend)\n")
        
        print("### Category Breakdown\n")
        for cat in sorted(categories.keys(), key=lambda x: categories[x], reverse=True):
            print(f"- **{cat}**: {categories[cat]} projects")

def main():
    """Interactive CLI"""
    expert = AwesomeLLMExpert()
    
    print("="*60)
    print("🤖 Awesome LLM Apps - Expert AI Agent Architect")
    print("="*60)
    print(f"\n✅ Loaded {len(expert.projects)} projects from repository\n")
    
    print("💡 Available Commands:")
    print("   1. analyze <project_name>  - Analyze a specific project")
    print("   2. list                    - List all categories")
    print("   3. compare <name1> <name2> - Compare two projects")
    print("   4. generate <domain>       - Generate new use case")
    print("   5. stats                   - Show repository statistics")
    print("   6. exit                    - Exit program\n")
    
    while True:
        try:
            command = input("🔍 Enter command: ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'exit' or cmd == 'quit':
                print("\n👋 Goodbye!")
                break
            elif cmd == 'analyze' and len(parts) > 1:
                expert.analyze_project(' '.join(parts[1:]))
            elif cmd == 'list':
                expert.list_categories()
            elif cmd == 'compare' and len(parts) >= 3:
                expert.compare_projects(parts[1], parts[2])
            elif cmd == 'generate' and len(parts) > 1:
                expert.generate_new_use_case(' '.join(parts[1:]))
            elif cmd == 'stats':
                expert.show_stats()
            else:
                print("❌ Invalid command. Try: analyze, list, compare, generate, stats, exit\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == '__main__':
    main()
