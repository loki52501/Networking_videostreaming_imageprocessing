# AVL Tree Build Plan

## Core Invariant
- Binary search tree property: left subtree < node < right subtree.
- Height balance: `|height(left) - height(right)| <= 1` for every node.

## Data Structure Sketch
```cpp
struct Node {
    int key;
    int height;              // recompute after changes
    Node* left{nullptr};
    Node* right{nullptr};
};
```
- Store `height` or `balance = height(left) - height(right)`; choose one.
- Provide helpers: `int height(Node*)`, `int balance(Node*)`, `void update(Node*)`.

## Operations to Implement
1. **Insert(key)**
   - Recurse as in BST insert; create new node at leaf.
   - On unwind: update `height`, compute balance, rotate if balance ? {+2, -2}.
   - Rotations:
     - LL ? single right rotation.
     - RR ? single left rotation.
     - LR ? left rotate child, then right rotate node.
     - RL ? right rotate child, then left rotate node.

2. **Erase(key)**
   - Standard BST deletion (replace with inorder successor when both children).
   - On unwind: update heights and rotate as in insert.
   - Consider helper `Node* erase(Node*, key)` to simplify recursion.

3. **Search(key)**
   - Plain BST walk.

4. **Invariants / Validation**
   - Function to compute `height` recursively and assert stored heights match.
   - Check balance condition while traversing.

## Testing Ideas
- Insert sorted ascending values to ensure rotations trigger.
- Insert random values and verify in-order traversal is sorted.
- Remove values in different orders; re-check invariants each time.
- Measure rotation counts to compare against naive BST if you want latency insight.

## Stretch Goals
- Store payload (`Value` template parameter) besides key.
- Add `lower_bound`/`upper_bound` functions.
- Track node count to support `select(k)` (k-th smallest) or `rank(key)`.
