"""
Tree Visualizer for RAPTOR System.

Provides multiple visualization methods:
- ASCII tree representation
- HTML interactive tree
- GraphML export for Neo4j Browser
- Statistics and analysis views
"""

from typing import Dict, List, Optional, Set, Tuple
from io import StringIO
import json
from loguru import logger

from .tree_builder import TreeNode, TreeStats


class TreeVisualizer:
    """
    Visualizes RAPTOR trees in various formats.

    Supports ASCII, HTML, and GraphML outputs.
    """

    def __init__(self, max_content_length: int = 50):
        """
        Initialize tree visualizer.

        Args:
            max_content_length: Maximum content length in display
        """
        self.max_content_length = max_content_length

        logger.info(f"TreeVisualizer initialized: max_content_length={max_content_length}")

    def visualize_ascii(
        self,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode],
        show_content: bool = True,
        show_scores: bool = True,
        max_depth: Optional[int] = None
    ) -> str:
        """
        Create ASCII tree representation.

        Args:
            root: Root node
            all_nodes: All nodes
            show_content: Show node content
            show_scores: Show quality scores
            max_depth: Maximum depth to show

        Returns:
            ASCII tree string
        """
        output = StringIO()

        def _render_node(
            node: TreeNode,
            prefix: str = "",
            is_last: bool = True,
            depth: int = 0
        ):
            if max_depth is not None and depth > max_depth:
                return

            # Prepare node label
            label_parts = [f"L{node.level}"]

            if node.is_leaf:
                label_parts.append("LEAF")
            else:
                label_parts.append(f"({len(node.children_ids)} children)")

            if show_scores and node.summary:
                label_parts.append(f"Q:{node.summary.quality_score:.2f}")

            label = " ".join(label_parts)

            # Prepare content preview
            if show_content:
                content_preview = node.content[:self.max_content_length]
                if len(node.content) > self.max_content_length:
                    content_preview += "..."
                content_preview = content_preview.replace("\n", " ")
            else:
                content_preview = ""

            # Build line
            connector = "└── " if is_last else "├── "
            line = f"{prefix}{connector}{label}"

            if content_preview:
                line += f": {content_preview}"

            output.write(line + "\n")

            # Prepare prefix for children
            if is_last:
                child_prefix = prefix + "    "
            else:
                child_prefix = prefix + "│   "

            # Render children
            for i, child_id in enumerate(node.children_ids):
                if child_id in all_nodes:
                    child = all_nodes[child_id]
                    is_last_child = (i == len(node.children_ids) - 1)
                    _render_node(child, child_prefix, is_last_child, depth + 1)

        # Start rendering from root
        output.write(f"Tree: {root.node_id}\n")
        output.write(f"{'=' * 60}\n")

        _render_node(root)

        return output.getvalue()

    def visualize_level(
        self,
        level: int,
        all_nodes: Dict[str, TreeNode],
        show_content: bool = True
    ) -> str:
        """
        Visualize all nodes at a specific level.

        Args:
            level: Tree level
            all_nodes: All nodes
            show_content: Show node content

        Returns:
            Level visualization
        """
        level_nodes = [n for n in all_nodes.values() if n.level == level]

        if not level_nodes:
            return f"No nodes at level {level}"

        output = StringIO()
        output.write(f"\nLevel {level} ({len(level_nodes)} nodes)\n")
        output.write(f"{'=' * 60}\n\n")

        for i, node in enumerate(level_nodes, 1):
            output.write(f"{i}. {node.node_id}\n")

            if show_content:
                content = node.content[:self.max_content_length * 2]
                if len(node.content) > self.max_content_length * 2:
                    content += "..."
                output.write(f"   {content}\n")

            output.write(f"   Children: {len(node.children_ids)}, ")
            output.write(f"Siblings: {len(node.sibling_ids)}\n")

            if node.summary:
                output.write(f"   Quality: {node.summary.quality_score:.3f}\n")

            output.write("\n")

        return output.getvalue()

    def visualize_path(
        self,
        node_id: str,
        all_nodes: Dict[str, TreeNode],
        show_content: bool = True
    ) -> str:
        """
        Visualize path from node to root.

        Args:
            node_id: Starting node ID
            all_nodes: All nodes
            show_content: Show node content

        Returns:
            Path visualization
        """
        if node_id not in all_nodes:
            return f"Node {node_id} not found"

        output = StringIO()
        output.write(f"\nPath to root from {node_id}\n")
        output.write(f"{'=' * 60}\n\n")

        current_id = node_id
        depth = 0

        while current_id and current_id in all_nodes:
            node = all_nodes[current_id]

            indent = "  " * depth
            output.write(f"{indent}Level {node.level}: {node.node_id}\n")

            if show_content:
                content = node.content[:self.max_content_length]
                if len(node.content) > self.max_content_length:
                    content += "..."
                output.write(f"{indent}  {content}\n")

            if node.parent_id:
                output.write(f"{indent}  ↓\n")

            current_id = node.parent_id
            depth += 1

        return output.getvalue()

    def visualize_stats(
        self,
        stats: TreeStats,
        all_nodes: Dict[str, TreeNode]
    ) -> str:
        """
        Visualize tree statistics.

        Args:
            stats: Tree statistics
            all_nodes: All nodes

        Returns:
            Statistics visualization
        """
        output = StringIO()

        output.write("\nTree Statistics\n")
        output.write(f"{'=' * 60}\n\n")

        output.write(f"Total Nodes:           {stats.total_nodes}\n")
        output.write(f"Leaf Nodes:            {stats.leaf_nodes}\n")
        output.write(f"Internal Nodes:        {stats.internal_nodes}\n")
        output.write(f"Maximum Depth:         {stats.max_depth}\n")
        output.write(f"Average Branching:     {stats.avg_branching_factor:.2f}\n")
        output.write(f"Number of Levels:      {stats.num_levels}\n\n")

        output.write("Nodes per Level:\n")
        for level in sorted(stats.nodes_per_level.keys()):
            count = stats.nodes_per_level[level]
            bar = "█" * min(count, 50)
            output.write(f"  Level {level:2d}: {count:4d} {bar}\n")

        output.write("\n")

        # Compute additional statistics
        leaf_nodes = [n for n in all_nodes.values() if n.is_leaf]
        internal_nodes = [n for n in all_nodes.values() if not n.is_leaf]

        if internal_nodes:
            avg_children = sum(len(n.children_ids) for n in internal_nodes) / len(internal_nodes)
            output.write(f"Avg Children per Parent: {avg_children:.2f}\n")

            # Quality scores
            quality_scores = [
                n.summary.quality_score
                for n in internal_nodes
                if n.summary
            ]

            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                min_quality = min(quality_scores)
                max_quality = max(quality_scores)

                output.write(f"\nSummary Quality:\n")
                output.write(f"  Average: {avg_quality:.3f}\n")
                output.write(f"  Min:     {min_quality:.3f}\n")
                output.write(f"  Max:     {max_quality:.3f}\n")

        return output.getvalue()

    def export_to_html(
        self,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode],
        output_path: str
    ) -> bool:
        """
        Export tree to interactive HTML.

        Args:
            root: Root node
            all_nodes: All nodes
            output_path: Output file path

        Returns:
            Success status
        """
        try:
            html_content = self._generate_html(root, all_nodes)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Exported tree to HTML: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export HTML: {e}")
            return False

    def _generate_html(
        self,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> str:
        """Generate HTML content for tree."""
        # Convert tree to JSON for JavaScript
        tree_data = self._tree_to_json(root, all_nodes)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAPTOR Tree Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
        }}
        .tree {{
            margin-top: 20px;
        }}
        .node {{
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #fafafa;
        }}
        .node.leaf {{
            background-color: #e8f5e9;
        }}
        .node.internal {{
            background-color: #e3f2fd;
        }}
        .node-header {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .node-content {{
            margin: 5px 0;
            color: #666;
        }}
        .node-meta {{
            font-size: 0.9em;
            color: #999;
        }}
        .children {{
            margin-left: 30px;
            border-left: 2px solid #ddd;
            padding-left: 10px;
        }}
        .stats {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .stats table {{
            width: 100%;
        }}
        .stats td {{
            padding: 5px;
        }}
        .stats td:first-child {{
            font-weight: bold;
            width: 200px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RAPTOR Tree Visualization</h1>

        <div class="stats">
            <h2>Tree Statistics</h2>
            <table>
                <tr>
                    <td>Total Nodes:</td>
                    <td id="total-nodes"></td>
                </tr>
                <tr>
                    <td>Leaf Nodes:</td>
                    <td id="leaf-nodes"></td>
                </tr>
                <tr>
                    <td>Maximum Depth:</td>
                    <td id="max-depth"></td>
                </tr>
            </table>
        </div>

        <div class="tree" id="tree-root"></div>
    </div>

    <script>
        const treeData = {json.dumps(tree_data, indent=2)};

        function renderNode(node, container) {{
            const nodeDiv = document.createElement('div');
            nodeDiv.className = 'node ' + (node.is_leaf ? 'leaf' : 'internal');

            const header = document.createElement('div');
            header.className = 'node-header';
            header.textContent = `Level ${{node.level}}: ${{node.node_id}}`;
            nodeDiv.appendChild(header);

            const content = document.createElement('div');
            content.className = 'node-content';
            content.textContent = node.content.substring(0, 100) + (node.content.length > 100 ? '...' : '');
            nodeDiv.appendChild(content);

            const meta = document.createElement('div');
            meta.className = 'node-meta';
            meta.textContent = `Children: ${{node.children ? node.children.length : 0}}`;
            if (node.quality_score) {{
                meta.textContent += ` | Quality: ${{node.quality_score.toFixed(3)}}`;
            }}
            nodeDiv.appendChild(meta);

            container.appendChild(nodeDiv);

            if (node.children && node.children.length > 0) {{
                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'children';
                node.children.forEach(child => renderNode(child, childrenDiv));
                container.appendChild(childrenDiv);
            }}
        }}

        function updateStats(data) {{
            document.getElementById('total-nodes').textContent = data.stats.total_nodes;
            document.getElementById('leaf-nodes').textContent = data.stats.leaf_nodes;
            document.getElementById('max-depth').textContent = data.stats.max_depth;
        }}

        const treeContainer = document.getElementById('tree-root');
        renderNode(treeData.tree, treeContainer);
        updateStats(treeData);
    </script>
</body>
</html>"""

        return html

    def _tree_to_json(
        self,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> Dict[str, Any]:
        """Convert tree to JSON structure."""
        def _node_to_dict(node: TreeNode) -> Dict[str, Any]:
            node_dict = {
                'node_id': node.node_id,
                'content': node.content,
                'level': node.level,
                'is_leaf': node.is_leaf,
                'children': []
            }

            if node.summary:
                node_dict['quality_score'] = node.summary.quality_score

            # Add children recursively
            for child_id in node.children_ids:
                if child_id in all_nodes:
                    child = all_nodes[child_id]
                    node_dict['children'].append(_node_to_dict(child))

            return node_dict

        # Compute stats
        leaf_count = sum(1 for n in all_nodes.values() if n.is_leaf)

        return {
            'tree': _node_to_dict(root),
            'stats': {
                'total_nodes': len(all_nodes),
                'leaf_nodes': leaf_count,
                'max_depth': root.level
            }
        }

    def export_to_graphml(
        self,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode],
        output_path: str
    ) -> bool:
        """
        Export tree to GraphML format for Neo4j Browser.

        Args:
            root: Root node
            all_nodes: All nodes
            output_path: Output file path

        Returns:
            Success status
        """
        try:
            graphml_content = self._generate_graphml(root, all_nodes)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(graphml_content)

            logger.info(f"Exported tree to GraphML: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export GraphML: {e}")
            return False

    def _generate_graphml(
        self,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> str:
        """Generate GraphML content."""
        output = StringIO()

        # Write header
        output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        output.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n')
        output.write('    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        output.write('    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n')
        output.write('    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')

        # Define attributes
        output.write('  <key id="content" for="node" attr.name="content" attr.type="string"/>\n')
        output.write('  <key id="level" for="node" attr.name="level" attr.type="int"/>\n')
        output.write('  <key id="is_leaf" for="node" attr.name="is_leaf" attr.type="boolean"/>\n')
        output.write('  <key id="type" for="edge" attr.name="type" attr.type="string"/>\n')

        # Start graph
        output.write('  <graph id="G" edgedefault="directed">\n')

        # Write nodes
        for node in all_nodes.values():
            output.write(f'    <node id="{node.node_id}">\n')
            output.write(f'      <data key="content">{self._escape_xml(node.content[:100])}</data>\n')
            output.write(f'      <data key="level">{node.level}</data>\n')
            output.write(f'      <data key="is_leaf">{str(node.is_leaf).lower()}</data>\n')
            output.write('    </node>\n')

        # Write edges
        edge_id = 0
        for node in all_nodes.values():
            # Parent-child edges
            if node.parent_id:
                output.write(f'    <edge id="e{edge_id}" source="{node.node_id}" target="{node.parent_id}">\n')
                output.write('      <data key="type">PARENT</data>\n')
                output.write('    </edge>\n')
                edge_id += 1

            # Sibling edges
            for sibling_id in node.sibling_ids:
                if node.node_id < sibling_id:  # Avoid duplicates
                    output.write(f'    <edge id="e{edge_id}" source="{node.node_id}" target="{sibling_id}">\n')
                    output.write('      <data key="type">SIBLING</data>\n')
                    output.write('    </edge>\n')
                    edge_id += 1

        # Close graph and graphml
        output.write('  </graph>\n')
        output.write('</graphml>\n')

        return output.getvalue()

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters."""
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&apos;')
        return text

    def find_nodes_by_content(
        self,
        all_nodes: Dict[str, TreeNode],
        search_term: str,
        case_sensitive: bool = False
    ) -> List[TreeNode]:
        """
        Find nodes containing search term.

        Args:
            all_nodes: All nodes
            search_term: Search term
            case_sensitive: Case sensitive search

        Returns:
            Matching nodes
        """
        if not case_sensitive:
            search_term = search_term.lower()

        matching_nodes = []

        for node in all_nodes.values():
            content = node.content if case_sensitive else node.content.lower()

            if search_term in content:
                matching_nodes.append(node)

        return matching_nodes
