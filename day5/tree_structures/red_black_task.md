# Red-Black Tree Build Plan

## Core Invariants
1. Every node is either red or black; root is always black.
2. Red nodes cannot have red children (no double-red).
3. Every path from a node to descendant NIL leaves has the same number of black nodes (black-height).
4. NIL leaves are treated as black sentinel nodes.

## Data Structure Sketch
```cpp
enum class Color { Red, Black };

struct Node {
    int key;
    Color color{Color::Red};
    Node* parent{nullptr};
    Node* left{nullptr};
    Node* right{nullptr};
};
```
- Consider using a single shared `Node* nil` sentinel instead of `nullptr` to simplify edge cases.
- Helper routines: `bool is_red(Node*)`, `void set_color(Node*, Color)`, `Node* sibling(Node*)`.

## Operations to Implement
1. **Insert(key)**
   - Standard BST insertion; new node starts red.
   - Fix-up loop while parent is red:
     - Case 1: parent’s sibling (uncle) is red ? recolor parent, uncle to black, grandparent to red, move up.
     - Case 2: triangle (node is right child of left parent or vice versa) ? rotate parent.
     - Case 3: line (node aligns with parent) ? rotate grandparent, swap colors of parent/grandparent.
   - Ensure root becomes black at end.

2. **Erase(key)**
   - Remove node like BST; if node or replacement is red, recolor black and stop.
   - Otherwise fix double-black:
     - Case 1: sibling is red ? rotate sibling toward node, recolor.
     - Case 2: sibling black with black children ? recolor sibling red, move up.
     - Case 3: sibling black with inner red child ? rotate sibling toward sibling, recolor.
     - Case 4: sibling black with outer red child ? rotate parent, recolor appropriately, clear double-black.
   - Keep NIL sentinel black to simplify checks.

3. **Search / Traversal**
   - Same as BST; use sentinel checks.

## Validation Tools
- Write `int black_height(Node*)` to verify equal black count on all paths.
- Confirm no two consecutive reds in DFS.
- Ensure root marked black after every operation.

## Testing Ideas
- Insert ordered sequences, random sequences; compare depth vs. AVL.
- Delete keys in random order; after each step run validation functions.
- Compare inorder output with std::set for correctness (same sorted sequence).

## Stretch Goals
- Template the key type with comparator.
- Implement iterators (in-order successor/predecessor).
- Add tree visualization (dot graph) to watch color rotations.
