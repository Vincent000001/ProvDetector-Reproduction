"""
Convert Linux .dot provenance graph files to CSV format
"""
import os
import re
import pandas as pd
import argparse
from pathlib import Path


def parse_dot_file(dot_file_path):
    """
    Parse a .dot file and convert to CSV format matching our data structure
    Handles both Linux and Windows .dot formats
    """
    edges = []
    
    with open(dot_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Pattern for edges: source -> dest [attributes]
    # Handle both quoted and unquoted node IDs, with or without attributes
    # Format: "node1" -> "node2" [attributes] or node1 -> node2 [attributes]
    # Escape the -> properly in the character class
    edge_pattern = r'(["\']?)([^"\'\s\-]+)\1\s*->\s*(["\']?)([^"\'\s\[\]]+)\3(?:\s*\[(.*?)\])?'
    
    # Extract all edges
    for match in re.finditer(edge_pattern, content, re.DOTALL):
        source = match.group(2).strip('"\'')
        dest = match.group(4).strip('"\'')
        attrs_str = match.group(5) if match.group(5) else ''
        
        # Parse attributes - handle both key="value" and key=value formats
        attrs = {}
        if attrs_str:
            # Try quoted values first
            for attr_match in re.finditer(r'(\w+)="([^"]*)"', attrs_str):
                key = attr_match.group(1)
                value = attr_match.group(2)
                attrs[key] = value
            
            # Also try unquoted values
            for attr_match in re.finditer(r'(\w+)=([^\s,\]]+)', attrs_str):
                key = attr_match.group(1)
                value = attr_match.group(2).strip('"\'')
                if key not in attrs:  # Don't overwrite quoted values
                    attrs[key] = value
        
        # Try to get node information from node definitions
        # Look for node definitions: "node_id" [label="...", type=...]
        node_info = {}
        node_pattern = rf'["\']?{re.escape(source)}["\']?\s*\[(.*?)\]'
        node_match = re.search(node_pattern, content, re.DOTALL)
        if node_match:
            node_attrs_str = node_match.group(1)
            for attr_match in re.finditer(r'(\w+)=([^\s,\]]+)', node_attrs_str):
                key = attr_match.group(1)
                value = attr_match.group(2).strip('"\'')
                node_info[key] = value
        
        # Extract information with defaults
        # Infer types from node labels/IDs
        source_type = attrs.get('sourceType', 'process')
        if '/' in source or source.startswith('/'):
            source_type = 'file'
        elif ':' in source or source.startswith(('192.', '10.', '172.')):
            source_type = 'socket'
            
        dest_type = attrs.get('destType', 'process')
        if '/' in dest or dest.startswith('/'):
            dest_type = 'file'
        elif ':' in dest or dest.startswith(('192.', '10.', '172.')):
            dest_type = 'socket'
        
        action = attrs.get('action', 'connect')
        # Default action based on types
        if source_type == 'process' and dest_type == 'process':
            action = 'start'
        elif dest_type == 'file':
            action = 'open'  # Linux uses 'open'
        elif dest_type == 'socket':
            action = 'connect'
            
        process_name = attrs.get('processName', node_info.get('label', source.split('/')[-1] if '/' in source else source))
        timestamp = attrs.get('timestamp', '0')
        pid0 = attrs.get('pid0', '')
        pid1 = attrs.get('pid1', '')  # May or may not exist
        
        edges.append({
            'sourceId': source,
            'sourceType': source_type,
            'destinationId': dest,
            'destinationType': dest_type,
            'action': action,
            'processName': process_name,
            'timestamp': timestamp,
            'pid0': pid0 if pid0 else '',
            'pid1': pid1 if pid1 else ''
        })
    
    return pd.DataFrame(edges)


def convert_dot_directory(dot_dir, output_dir):
    """Convert all .dot files in a directory to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    dot_files = list(Path(dot_dir).rglob('*.dot'))
    print(f"Found {len(dot_files)} .dot files")
    
    for dot_file in dot_files:
        try:
            df = parse_dot_file(str(dot_file))
            if not df.empty:
                # Create output filename
                output_name = dot_file.stem + '.csv'
                output_path = os.path.join(output_dir, output_name)
                df.to_csv(output_path, index=False)
                print(f"Converted: {dot_file.name} -> {output_name} ({len(df)} edges)")
            else:
                print(f"Warning: {dot_file.name} produced empty CSV")
        except Exception as e:
            print(f"Error converting {dot_file.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert Linux .dot files to CSV')
    parser.add_argument('--dot-dir', type=str, required=True,
                       help='Directory containing .dot files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Linux .dot to CSV Converter")
    print("=" * 60)
    print(f"Input directory: {args.dot_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    convert_dot_directory(args.dot_dir, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

