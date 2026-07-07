class Node:
    """
    A class to represent a node in the linked list.
    Attributes:
        data: The data stored in the node
        next: Reference to the next node in the list
    """
    def __init__(self, data):
        self.data = data
        self.next = None
    
class LinkedListException(Exception):
    """Custom exception class for LinkedList operations."""
    pass

class LinkedList:
    """
    A class to implement a singly linked list using OOP principles.
    Attributes:
        head: Reference to the first node in the list
        size: Number of nodes in the list
    """  
    def __init__(self):
        #Initialize an empty linked list.
        self.head = None
        self.size = 0
    
    def is_empty(self):
        """
        Check if the linked list is empty.
        return True if the list is empty, False otherwise
        """
        return self.head is None
    
    def get_size(self):
        #Get the size of the linked list.
        return self.size
    
    def append(self, data):
        #Add a new node with the given data to the end of the list.
        new_node = Node(data)
        
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self.size += 1
        print(f"Added '{data}' to the list")  
  
    def print_list(self):
        #Print all elements in the linked list.
        if self.is_empty():
            print("The list is empty")
            return
        
        current = self.head
        elements = []
        
        while current:
            elements.append(str(current.data))
            current = current.next
        
        print("Linked List: " + " -> ".join(elements) + " -> None")
        print(f"Size: {self.size}")
    
    def delete_nth_node(self, n):
        """
        Delete the nth node from the linked list (1-based indexing).
        
        Args:
            n (int): The position of the node to delete (1-based)
            
        Raises:
            LinkedListException: If the list is empty or index is out of range
        """
        # Validate input
        if self.is_empty():
            raise LinkedListException("Cannot delete from an empty list")
        
        if n < 1:
            raise LinkedListException("Index must be a positive integer (1-based)")
        
        if n > self.size:
            raise LinkedListException(f"Index {n} is out of range. List has only {self.size} elements")
        
        # Special case: delete the first node
        if n == 1:
            deleted_data = self.head.data
            self.head = self.head.next
            self.size -= 1
            print(f"Deleted node at position {n} with data '{deleted_data}'")
            return deleted_data
        
        # Find the node before the one to delete
        current = self.head
        for i in range(1, n - 1):
            current = current.next
        
        # Store the data of the node to be deleted
        deleted_data = current.next.data
        
        # Remove the node by updating the link
        current.next = current.next.next
        self.size -= 1
        
        print(f"Deleted node at position {n} with data '{deleted_data}'")
        return deleted_data
    
    def find(self, data):
        #Find the position of the first occurrence of data in the list.
        current = self.head
        position = 1
        
        while current:
            if current.data == data:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def delete_by_value(self, data):
        """
        Delete the first node with the specified data.
        
        Args:
            data: The data to delete
            
        Returns:
            bool: True if deletion was successful, False if data not found
        """
        position = self.find(data)
        if position == -1:
            print(f"Data '{data}' not found in the list")
            return False
        
        try:
            self.delete_nth_node(position)
            return True
        except LinkedListException as e:
            print(f"Error: {e}")
            return False


def test_linked_list():
    """Test the LinkedList implementation with various scenarios."""
    print("=" * 60)
    print("           LINKED LIST IMPLEMENTATION TEST")
    print("=" * 60)
    
    # Create a new linked list
    ll = LinkedList()
    
    print("\n1. Testing with empty list:")
    ll.print_list()
    
    # Test deleting from empty list
    try:
        ll.delete_nth_node(1)
    except LinkedListException as e:
        print(f"Expected error: {e}")
    
    print("\n2. Adding elements to the list:")
    ll.append("Apple")
    ll.append("Banana")
    ll.append("Cherry")
    ll.append("Date")
    
    print("\n3. Current list:")
    ll.print_list()
    
    print("\n4. Testing deletion of nth nodes:")
    
    # Delete first node
    try:
        ll.delete_nth_node(1)
        ll.print_list()
    except LinkedListException as e:
        print(f"Error: {e}")
    
    # Delete middle node
    try:
        ll.delete_nth_node(2)
        ll.print_list()
    except LinkedListException as e:
        print(f"Error: {e}")
    
    # Delete last node
    try:
        ll.delete_nth_node(ll.get_size())
        ll.print_list()
    except LinkedListException as e:
        print(f"Error: {e}")
    
    print("\n5. Testing edge cases:")
    
    # Try to delete with invalid index
    try:
        ll.delete_nth_node(0)
    except LinkedListException as e:
        print(f"Expected error: {e}")
    
    try:
        ll.delete_nth_node(10)
    except LinkedListException as e:
        print(f"Expected error: {e}")
    
    print("\n6. Testing find and delete by value:")
    ll.append("Elderberry")
    ll.append("Fig")
    ll.print_list()
    
    position = ll.find("Cherry")
    if position != -1:
        print(f"Found 'Cherry' at position {position}")
    else:
        print("'Cherry' not found")
    
    ll.delete_by_value("Cherry")
    ll.print_list()
    
    print("\n7. Final list state:")
    ll.print_list()
    
# Run the test
if __name__ == "__main__":
    test_linked_list()
