"""
Elementary data structures built from scratch.

  DynamicArray        - resizable array on a fixed-size block (doubling)
  Matrix              - 2-D array stored row by row in one flat block
  ArrayStack          - stack on a DynamicArray (top = end)
  ArrayQueue          - circular-buffer queue on a fixed block, grows by doubling
  ShiftingArrayQueue  - naive array queue that shifts on dequeue (for contrast)
  SinglyLinkedList    - head + tail pointers, insertion / deletion / traversal
  LinkedStack, LinkedQueue - stack and queue on linked nodes (for comparison)
  RootedTree          - left-child, right-sibling linked representation

"""


def _block(capacity):
    """A fixed-size block of memory slots."""
    return [None] * capacity


# =========================================================================
# Arrays
# =========================================================================
class DynamicArray:
    """Resizable array. Access O(1); append amortized O(1);
    insert/delete at index i O(n - i)."""

    def __init__(self, capacity=4):
        self._cap = max(1, capacity)     # number of slots in the block
        self._data = _block(self._cap)   # the block itself
        self._n = 0                      # slots actually in use: a[0..n-1]

    def __len__(self):
        return self._n

    def _check(self, i):
        # Only positions 0..n-1 hold real data; the rest of the block is
        # spare capacity and must not be readable from outside.
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range for length {self._n}")

    def _resize(self, new_cap):
        # Allocate a fresh block and copy the live elements across: O(n).
        # Because capacity doubles each time, these copies add up to at most
        # about 2n element moves over n appends, i.e. O(1) amortized each.
        new = _block(new_cap)
        for i in range(self._n):
            new[i] = self._data[i]
        self._data, self._cap = new, new_cap

    # access ---------------------------------------------------------
    def get(self, i):
        self._check(i)
        return self._data[i]

    def set(self, i, value):
        self._check(i)
        self._data[i] = value

    __getitem__ = get
    __setitem__ = set

    # insertion ------------------------------------------------------
    def append(self, value):
        if self._n == self._cap:
            self._resize(2 * self._cap)          # doubling => amortized O(1)
        self._data[self._n] = value                # place the new value at the end of the array
        self._n += 1

    def insert(self, i, value):
        if not 0 <= i <= self._n:
            raise IndexError(f"insert position {i} out of range")
        if self._n == self._cap:
            self._resize(2 * self._cap)         # ensure there is enough space for the new element
        for j in range(self._n, i, -1):          # shift right
            self._data[j] = self._data[j - 1]
        self._data[i] = value                   # place the new value at the specified position
        self._n += 1

    # deletion -------------------------------------------------------
    def delete(self, i):
        self._check(i)
        value = self._data[i]
        for j in range(i, self._n - 1):          
            self._data[j] = self._data[j + 1]       # shift each element one position to the left
        self._n -= 1
        self._data[self._n] = None               # drop the reference
        """Shrink at 1/4 full (not 1/2) so that alternating append/delete at
        the boundary cannot trigger a resize on every single call."""
        if 0 < self._n <= self._cap // 4 and self._cap > 4:
            self._resize(self._cap // 2)         # shrink to save memory
        return value

    def pop(self):
        if self._n == 0:
            raise IndexError("pop from empty array")
        return self.delete(self._n - 1)

    # search / traversal ----------------------------------------------
    def index(self, value):
        for i in range(self._n):
            if self._data[i] == value:
                return i                    # return the index of the found value
        return -1

    def __iter__(self):
        for i in range(self._n):
            yield self._data[i]               # yield each element in order

    def __repr__(self):
        return f"DynamicArray({list(self)})"


class Matrix:
    """rows x cols matrix stored row-major in one flat block:
    element (r, c) lives at r * cols + c, so access is O(1)."""

    def __init__(self, rows, cols, fill=0):
        if rows < 0 or cols < 0:
            raise ValueError("dimensions must be non-negative")
        self.rows, self.cols = rows, cols
        self._data = _block(rows * cols)       # allocate a flat block for the matrix elements
        for k in range(rows * cols):
            self._data[k] = fill        # initialize each element to the fill value

    @classmethod
    def from_rows(cls, rows):
        m = cls(len(rows), len(rows[0]) if rows else 0)
        for r, row in enumerate(rows):
            if len(row) != m.cols:
                raise ValueError("ragged rows")         # ensure all rows have the same number of columns
            for c, v in enumerate(row):
                m.set(r, c, v)
        return m

    def _pos(self, r, c):
        # Row-major layout: row r starts at r * cols, so (r, c) maps to a
        # single flat index. This is how C and NumPy lay out 2-D arrays.
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError(f"({r}, {c}) out of range for {self.rows}x{self.cols}")
        return r * self.cols + c

    def get(self, r, c):
        return self._data[self._pos(r, c)]       # return the element at the specified position

    def set(self, r, c, v):
        self._data[self._pos(r, c)] = v

    def insert_row(self, r, values):
        """Insert a row before row r. O(rows * cols) because the block is rebuilt."""
        if not 0 <= r <= self.rows or len(values) != self.cols:
            raise ValueError("bad row position or length")
        new = _block((self.rows + 1) * self.cols)
        for i in range(self.rows + 1):
            for c in range(self.cols):
                if i < r:
                    new[i * self.cols + c] = self._data[i * self.cols + c] # copy the existing row before the insertion point
                elif i == r:
                    new[i * self.cols + c] = values[c]       # insert the new row at the specified position
                else:
                    new[i * self.cols + c] = self._data[(i - 1) * self.cols + c]
        self._data, self.rows = new, self.rows + 1

    def delete_row(self, r):
        if not 0 <= r < self.rows:
            raise IndexError("row out of range")
        new = _block((self.rows - 1) * self.cols)
        k = 0
        for i in range(self.rows):
            if i == r:
                continue                    # skip the row to be deleted
            for c in range(self.cols):
                new[k] = self._data[i * self.cols + c]  # copy the element from the original matrix to the new matrix
                k += 1
        self._data, self.rows = new, self.rows - 1

    def insert_col(self, c, values):
        if not 0 <= c <= self.cols or len(values) != self.rows:
            raise ValueError("bad column position or length")
        nc = self.cols + 1
        new = _block(self.rows * nc)
        for r in range(self.rows):
            for j in range(nc):
                if j < c:
                    new[r * nc + j] = self._data[r * self.cols + j] # copy the existing column before the insertion point
                elif j == c:
                    new[r * nc + j] = values[r]       # insert the new column at the specified position
                else:
                    new[r * nc + j] = self._data[r * self.cols + j - 1]
        self._data, self.cols = new, nc

    def delete_col(self, c):
        if not 0 <= c < self.cols:
            raise IndexError("column out of range")
        nc = self.cols - 1
        new = _block(self.rows * nc)                # allocate a flat block for the new matrix with one fewer column
        for r in range(self.rows):
            k = 0
            for j in range(self.cols):
                if j != c:
                    new[r * nc + k] = self._data[r * self.cols + j]  # copy the element from the original matrix to the new matrix
                    k += 1
        self._data, self.cols = new, nc

    def transpose(self):
        t = Matrix(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                t.set(c, r, self.get(r, c)) # set the element in the transposed matrix at the swapped position
        return t

    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("inner dimensions differ")
        # Triple loop: entry (i, j) is the dot product of row i of
        # self with column j of other. Cost O(rows * cols * other.cols).
        out = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0
                for k in range(self.cols):
                    s += self.get(i, k) * other.get(k, j) # accumulate the product of corresponding elements from row i of self and column j of other
                out.set(i, j, s)
        return out

    def to_rows(self):
        return [[self.get(r, c) for c in range(self.cols)] for r in range(self.rows)] # convert the matrix to a list of rows


# =========================================================================
# Stacks and queues on arrays
# =========================================================================
class ArrayStack:
    """LIFO stack; the top is the end of a DynamicArray so push/pop are
    amortized O(1)."""

    def __init__(self):
        self._a = DynamicArray()

    def push(self, x):
        self._a.append(x)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._a.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._a.get(len(self._a) - 1)

    def is_empty(self):
        return len(self._a) == 0

    def __len__(self):
        return len(self._a)


class ArrayQueue:
    """FIFO queue as a circular buffer: head index wraps around the block,
    so enqueue and dequeue never shift elements. Amortized O(1)."""

    def __init__(self, capacity=4):
        self._cap = max(1, capacity)
        self._data = _block(self._cap)
        self._head = 0      # index of the front element
        self._n = 0
        # The back of the queue is not stored: it is always
        # (head + n) % cap, the first free slot after the last element.

    def _grow(self):
        # The live elements may wrap past the end of the block, so copy them
        # in queue order (front first) and restart the head at index 0.
        new = _block(2 * self._cap)
        for i in range(self._n):                 # unwrap into the new block
            new[i] = self._data[(self._head + i) % self._cap]   # copy the element from the old block to the new block in queue order
        self._data, self._cap, self._head = new, 2 * self._cap, 0

    def enqueue(self, x):
        if self._n == self._cap:
            self._grow()
        self._data[(self._head + self._n) % self._cap] = x  # place the new element at the back of the queue
        self._n += 1

    def dequeue(self):
        if self._n == 0:
            raise IndexError("dequeue from empty queue")
        x = self._data[self._head]
        self._data[self._head] = None            # let the item be freed
        # Instead of shifting every element left (O(n)), just move the
        # front marker forward, wrapping to 0 at the end of the block.
        self._head = (self._head + 1) % self._cap
        self._n -= 1
        return x

    def front(self):
        if self._n == 0:
            raise IndexError("front of empty queue")
        return self._data[self._head]

    def is_empty(self):
        return self._n == 0

    def __len__(self):
        return self._n


class ShiftingArrayQueue:
    """Naive array queue: dequeue removes index 0 and shifts everything
    left, so it costs O(n). Included only to show why the circular buffer
    matters."""

    def __init__(self):
        self._a = DynamicArray()

    def enqueue(self, x):
        self._a.append(x)

    def dequeue(self):
        if len(self._a) == 0:
            raise IndexError("dequeue from empty queue")
        return self._a.delete(0)

    def is_empty(self):
        return len(self._a) == 0

    def __len__(self):
        return len(self._a)


# =========================================================================
# Singly linked list
# =========================================================================
class Node:
    # __slots__ stops Python giving every node its own attribute dictionary,
    # which roughly halves the memory each node needs.
    __slots__ = ("value", "next")

    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class SinglyLinkedList:
    """Singly linked list with head and tail pointers.
    insert_front / insert_back / delete_front: O(1)
    insert_at / delete_at / delete_value / search: O(n)"""

    def __init__(self, values=()):
        self.head = None
        self.tail = None
        self._n = 0
        for v in values:
            self.insert_back(v)

    def __len__(self):
        return self._n

    # insertion ------------------------------------------------------
    def insert_front(self, value):
        # The new node points at the old head and becomes the head: O(1).
        self.head = Node(value, self.head)
        if self.tail is None:            # list was empty: node is also the tail
            self.tail = self.head
        self._n += 1

    def insert_back(self, value):
        # Keeping a tail pointer makes this O(1); without it we would have to
        # walk the whole list to find the last node.
        node = Node(value)
        if self.tail is None:
            self.head = self.tail = node # if the list was empty, the new node is both the head and the tail
        else:
            self.tail.next = node
            self.tail = node
        self._n += 1

    def insert_after(self, node, value):
        """O(1) once you already hold the node."""
        new = Node(value, node.next)
        node.next = new  # link the new node after the specified node
        if node is self.tail:
            self.tail = new  # update the tail if the new node is inserted at the end
        self._n += 1
        return new

    def insert_at(self, i, value):
        if not 0 <= i <= self._n:
            raise IndexError(f"insert position {i} out of range")
        if i == 0:
            return self.insert_front(value) # insert at the front if the index is 0
        if i == self._n:
            return self.insert_back(value) # insert at the back if the index is the last position
        # A singly linked node only knows its successor, so we must stop at
        # the node *before* position i and link the new node after it.
        self.insert_after(self._node_at(i - 1), value)

    # deletion -------------------------------------------------------
    def delete_front(self):
        if self.head is None:
            raise IndexError("delete from empty list")
        node = self.head
        self.head = node.next  # move the head to the next node
        if self.head is None:  # if the list is now empty
            self.tail = None  # clear the tail pointer
        self._n -= 1
        return node.value

    def delete_at(self, i):
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range")
        if i == 0:
            return self.delete_front()  # delete the front node if the index is 0
        prev = self._node_at(i - 1)  # get the node before the one to delete
        return self._unlink_after(prev)  # unlink the node after prev

    def delete_value(self, value):
        """Remove the first node holding value. Returns True if found."""
        if self.head is None:
            return False
        if self.head.value == value:
            self.delete_front()
            return True
        prev = self.head  # start from the head and look for the node to delete
        while prev.next is not None:  # iterate through the list until the last node
            if prev.next.value == value:  
                self._unlink_after(prev)  # unlink the node after prev
                return True  # return True if the value was found and deleted
            prev = prev.next
        return False 

    def _unlink_after(self, prev):
        # Bypass the node after prev. Nothing else points to it, so Python's
        # garbage collector reclaims it automatically.
        node = prev.next
        prev.next = node.next # bypass the node to be deleted
        if node is self.tail:
            self.tail = prev
        self._n -= 1
        return node.value

    # access / search / traversal ------------------------------------
    def _node_at(self, i):
        node = self.head
        for _ in range(i):
            node = node.next # move to the next node in the list
        return node

    def get(self, i):
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range")
        return self._node_at(i).value

    def find(self, value):
        """Return the index of the first match, or -1."""
        node, i = self.head, 0
        while node is not None:
            if node.value == value:
                return i
            node, i = node.next, i + 1 # move to the next node and increment the index
        return -1

    def traverse(self, visit):
        node = self.head
        while node is not None:
            visit(node.value)
            node = node.next

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node.value
            node = node.next

    def reverse(self):
        # Walk the list once, turning each `next` pointer to face backwards.
        # O(n) time, O(1) extra space; the old head becomes the new tail.
        prev, node = None, self.head
        self.tail = self.head
        while node is not None:
            node.next, prev, node = prev, node, node.next
        self.head = prev

    def __repr__(self):
        return " -> ".join(map(repr, self)) or "(empty)"


class LinkedStack:
    """Stack on linked nodes: push/pop at the head, worst-case O(1)."""

    def __init__(self):
        self._head = None
        self._n = 0

    def push(self, x):
        self._head = Node(x, self._head)
        self._n += 1

    def pop(self):
        if self._head is None:
            raise IndexError("pop from empty stack")
        x = self._head.value
        self._head = self._head.next
        self._n -= 1
        return x

    def peek(self):
        if self._head is None:
            raise IndexError("peek at empty stack")
        return self._head.value

    def is_empty(self):
        return self._head is None

    def __len__(self):
        return self._n


class LinkedQueue:
    """Queue on linked nodes: enqueue at tail, dequeue at head, worst-case O(1)."""

    def __init__(self):
        self._head = self._tail = None
        self._n = 0

    def enqueue(self, x):
        node = Node(x)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._n += 1

    def dequeue(self):
        if self._head is None:
            raise IndexError("dequeue from empty queue")
        x = self._head.value
        self._head = self._head.next
        if self._head is None:
            self._tail = None
        self._n -= 1
        return x

    def is_empty(self):
        return self._head is None

    def __len__(self):
        return self._n


# =========================================================================
# Rooted tree (left-child, right-sibling)
# =========================================================================
class TreeNode:
    __slots__ = ("value", "parent", "left_child", "right_sibling")

    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.left_child = None       # first child
        self.right_sibling = None    # next child of the same parent


class RootedTree:
    """A tree with any number of children per node, using only three
    pointers per node (parent, first child, next sibling)."""

    def __init__(self, root_value):
        self.root = TreeNode(root_value)
        self._n = 1

    def __len__(self):
        return self._n

    def add_child(self, parent, value):
        """Add a child as the new first child of parent: O(1)."""
        child = TreeNode(value, parent)
        child.right_sibling = parent.left_child         # link the new child to the current first child
        parent.left_child = child       # update the parent's first child to the new child
        self._n += 1
        return child

    @staticmethod
    def children(node):
        c = node.left_child
        while c is not None:
            yield c         # yield the current child node
            c = c.right_sibling

    def remove_subtree(self, node):
        """Detach node (and everything below it) from its parent: O(siblings)."""
        if node is self.root:
            raise ValueError("cannot remove the root")
        p = node.parent
        if p.left_child is node:
            p.left_child = node.right_sibling  # update the parent's first child to the next sibling of the node being removed
        else:
            c = p.left_child
            while c.right_sibling is not node:  # find the sibling just before the node to be removed
                c = c.right_sibling
            c.right_sibling = node.right_sibling
        self._n -= sum(1 for _ in self._preorder(node))
        node.parent = node.right_sibling = None

    def _preorder(self, node):
        # Depth-first: visit a node, then its children. Using an explicit
        # stack instead of recursion avoids Python's recursion limit on very
        # deep trees.
        stack = ArrayStack()                 # our own stack, no recursion
        stack.push(node)
        while not stack.is_empty():
            n = stack.pop()
            yield n
            kids = list(self.children(n))
            for k in reversed(kids):         # so the first child comes out first
                stack.push(k)

    def preorder(self):
        return [n.value for n in self._preorder(self.root)]

    def level_order(self):
        # Breadth-first: a FIFO queue makes every node at depth d come out
        # before any node at depth d + 1.
        q = ArrayQueue()                     # our own queue
        q.enqueue(self.root)
        out = []
        while not q.is_empty():
            n = q.dequeue()     # dequeue the next node from the queue
            out.append(n.value)
            for c in self.children(n):
                q.enqueue(c)
        return out

    def depth(self, node):
        d = 0
        while node.parent is not None:
            node, d = node.parent, d + 1 # move up to the parent and increment the depth
        return d

    def height(self):
        best = 0
        q = ArrayQueue()
        q.enqueue((self.root, 0))
        while not q.is_empty():
            n, d = q.dequeue()      # dequeue the next node and its depth from the queue
            best = max(best, d)
            for c in self.children(n):
                q.enqueue((c, d + 1))
        return best


# =========================================================================
# Self-test: written using AI, checked for accuracy
# =========================================================================
if __name__ == "__main__":
    import random
    rnd = random.Random(1)

    # DynamicArray against a Python list as reference
    a, ref = DynamicArray(), []
    for _ in range(5000):
        op = rnd.random()
        if op < 0.4 or not ref:
            v = rnd.randint(0, 99); a.append(v); ref.append(v)
        elif op < 0.6:
            i = rnd.randint(0, len(ref)); v = rnd.randint(0, 99)
            a.insert(i, v); ref.insert(i, v)
        elif op < 0.8:
            i = rnd.randrange(len(ref)); assert a.delete(i) == ref.pop(i)
        else:
            i = rnd.randrange(len(ref)); assert a.get(i) == ref[i]
        assert len(a) == len(ref)
    assert list(a) == ref

    # Matrix
    m = Matrix.from_rows([[1, 2, 3], [4, 5, 6]])
    assert m.transpose().to_rows() == [[1, 4], [2, 5], [3, 6]]
    assert m.multiply(m.transpose()).to_rows() == [[14, 32], [32, 77]]
    m.insert_row(1, [7, 8, 9]); assert m.to_rows() == [[1, 2, 3], [7, 8, 9], [4, 5, 6]]
    m.delete_col(0); assert m.to_rows() == [[2, 3], [8, 9], [5, 6]]
    m.insert_col(2, [0, 0, 0]); assert m.to_rows() == [[2, 3, 0], [8, 9, 0], [5, 6, 0]]
    m.delete_row(2); assert m.to_rows() == [[2, 3, 0], [8, 9, 0]]

    # Stacks and queues against reference behaviour
    for S in (ArrayStack, LinkedStack):
        s, ref = S(), []
        for _ in range(5000):
            if rnd.random() < 0.55 or not ref:
                v = rnd.random(); s.push(v); ref.append(v)
            else:
                assert s.pop() == ref.pop()
        assert len(s) == len(ref)
    from collections import deque
    for Q in (ArrayQueue, LinkedQueue, ShiftingArrayQueue):
        q, ref = Q(), deque()
        for _ in range(5000):
            if rnd.random() < 0.55 or not ref:
                v = rnd.random(); q.enqueue(v); ref.append(v)
            else:
                assert q.dequeue() == ref.popleft()
        assert len(q) == len(ref)

    # Linked list against a Python list
    ll, ref = SinglyLinkedList(), []
    for _ in range(3000):
        op = rnd.random()
        if op < 0.25 or not ref:
            v = rnd.randint(0, 50); ll.insert_front(v); ref.insert(0, v)
        elif op < 0.45:
            v = rnd.randint(0, 50); ll.insert_back(v); ref.append(v)
        elif op < 0.6:
            i = rnd.randint(0, len(ref)); v = rnd.randint(0, 50)
            ll.insert_at(i, v); ref.insert(i, v)
        elif op < 0.75:
            i = rnd.randrange(len(ref)); assert ll.delete_at(i) == ref.pop(i)
        elif op < 0.9:
            v = rnd.randint(0, 50)
            assert ll.delete_value(v) == (v in ref)
            if v in ref: ref.remove(v)
        else:
            v = rnd.randint(0, 50)
            assert ll.find(v) == (ref.index(v) if v in ref else -1)
        assert list(ll) == ref and len(ll) == len(ref)
        assert (ll.tail.value if ll.tail else None) == (ref[-1] if ref else None)
    ll.reverse(); assert list(ll) == ref[::-1]

    # Rooted tree
    t = RootedTree("A")
    b = t.add_child(t.root, "B"); c = t.add_child(t.root, "C")  # C becomes first child
    t.add_child(b, "D"); e = t.add_child(c, "E"); t.add_child(e, "F")
    assert t.preorder() == ["A", "C", "E", "F", "B", "D"]
    assert t.level_order() == ["A", "C", "B", "E", "D", "F"]
    assert t.height() == 3 and t.depth(e) == 2 and len(t) == 6
    t.remove_subtree(c); assert t.preorder() == ["A", "B", "D"] and len(t) == 3

    # Error handling
    for f in (lambda: ArrayStack().pop(), lambda: ArrayQueue().dequeue(),
              lambda: SinglyLinkedList().delete_front(), lambda: DynamicArray().get(0),
              lambda: Matrix(2, 2).get(2, 0)):
        try:
            f(); raise AssertionError("expected IndexError")
        except IndexError:
            pass
    print("All data structure tests passed.")