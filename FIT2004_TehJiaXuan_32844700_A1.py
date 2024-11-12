"""
FIT 2004 Assignment 1
Name: Teh Jia Xuan
Student ID: 32844700
"""

# ==========
# Q1


def fuse(fitmons):
    """
    This function fuse every combination of fitmons within that fitmons list
    to get the maximum cuteness of fitmon. This function iterate through all combination of fitmons
    by using a 2D array N x N where N is the number of fitmons. Every fitmon only able to fuse with the fitmons
    next to it. So this causes different combination of fitmons has different cuteness. This function
    is to find out the maximum cuteness of the final fitmons it can achieve. All fitmons must be fuse into one
    to get the final fitmons 

    Precondition: The input list (fitmons) must contain least 2 fitmons 
    Postcondition: return integer represent the max cuteness for every possible combination of final fitmons
                   all fitmons must be fused into final fitmon with every possible combination.
    Input:
        fitmons: a list of fitmons with [affiliaty_left, cuteness, affiliaty_right]
    Return:
        cuteness_score: integer represent max cuteness score for every possible combination of final fitmon

    Time complexity: 
        Best case analysis: O(N^2 + N^3) = O(N^3) where N is the number of fitmons
        Worst case analysis: O(N^3) where n is the number of fitmons as the algorithm 
                            needs to calculate every possible combination to achieve the cutest final fitmon

    Space complexity: 
        Input space analysis: O(N * 3)  = O(N) where n is the number of fitmons and each fitmons comes with 3 elements
        Aux space analysis:: O(N ^ 2) where n is the number of fitmons as having a 2D array to store all the fitmons
                             and the top right corner is the final fitmon with max cuteness
    """
    row = []
    memo = []
    result = 0
    
    for i in range(len(fitmons)): #O(n^2) time comp where n is number of fitmons 
        row = []                  #making 2D array nxn where n is number of fitmons
        for j in range(len(fitmons)):
            row.append(None)      #append none in row list
        memo.append(row)
    
    #overall time comp = O(n^3) where n is the number of fitmons
    for i in range(len(memo), -1, -1): # O(n^2/4) just traverse through half of the 2d array list 
        for j in range(i, len(memo)):  # where n is the number of fitmons
            max_cuteness = 0           
            if i == j:
                memo[i][j] = fitmons[i] # if is a diagonal then put it same as fitmons[i]
            else:
                for k in range(len(memo)): # O(n) where n is the number of fitmons
                    if i + k < j:          # to avoid NoneType error as it will calculate the only existing fitmons in the 2D array to get the new fitmons
                        fitmons_left = memo[i][i + k][2] * memo[i][i + k][1]  #get every position combination of fusing 
                        fitmons_right = memo[i+1+k][j][0] * memo[i+1+k][j][1] #k will increment every round 
                                                                              #so it moves to every possible combination of fitmons
                        max_cuteness = max(int(fitmons_left + fitmons_right), max_cuteness)      #get the max cuteness for every combination         
                        fitmons_affi_left = memo[i][i+k][0]                   #calculate the affi
                        fitmons_affi_right = memo[i+1+k][j][2]                 

                memo[i][j] = [fitmons_affi_left, max_cuteness, fitmons_affi_right] #place the maximum cuteness fitmon in the 2D array

    result = memo[0][len(memo) - 1][1]  # the top right corner is always the result
  
    return int(result)

# ==========
# Q2

class TreeMap:
    """
    A tree map graph, construct a graph according to the roads and solulu trees in the delulu forest
    """

    def __init__(self, roads, solulus):
        """
        Initialise the graph with given roads and solulus to construct delulu forest
        iterate through roads and solulus to set

        Precondition: roads is a list of tuple contain edges and weight from tree to tree
                      example (1,0,3) represent 1 -- 3 --> 0, tree 1 to 0 needs 3 minutes

                      solulus is a list of tuple contain solulu tree the tuple states which tree
                      it can teleport to and time claw. example (1,10,3) tree 1 teleport to tree 3 
                      need 10 minutes

        Postcondition: The delulu forest is in the TreeMap object with the given roads and solulus

        Input:
            roads: roads is a list of tuple contain edges and weight from tree to tree
                   example (1,0,3) represent 1 -- 3 --> 0, tree 1 to 0 needs 3 minutes

            solulus: solulus is a list of tuple contain solulu tree the tuple states which tree
                     it can teleport to and time claw. example (1,10,3) tree 1 teleport to tree 3 
                     need 10 minutes
        Return:
            None

        Time complexity: 
            Best case analysis: O(|T| + 2|R|) = O(|T| + |R|) where T is a set of unique trees of the graph
                                and R is the number of roads

            Worst case analysis: O(|T| + 2|R|) = O(|T| + |R|) where T is a set of unique trees of the graph
                                and R is the number of roads. We need to iterate through the road to find out
                                how many tree vertex are there so is R and we need to create a tree graph with
                                number of unique tree so is T. We need to iterate through all the tree vertex to 
                                add edges so is R = O(T + 2R) = O(T + R)

        Space complexity: 
            Input space analysis: O(|T| + |R|) where T is the number of solulus tree and R is 
                                  the number of roads

            Aux space analysis::  O(|T| + |R|) where T is the number of unique trees and R is 
                                  the number of roads. Creating a graph is T space comp and appending
                                  R number of roads to the tree vertex is R space comp so is O(T + R)
        """
        max_tree = 0
        self.roads = roads
        self.solulu_tree = solulus
        #O(R) time comp where R is the number of roads
        #O(1) space comp
        #to get the maximum node to construct graph
        for i in roads:
            max_tree = max(max_tree, i[1])
        
        #O(T) time comp where T set of unique trees
        #O(T) space comp where T is set of unique trees
        self.trees = Graph(max_tree + 1) #creating T vertex

        #O(R) time comp where R is the number of roads
        #O(R) space comp where R is the number of roads
        #add all the roads to the vertex's edge
        for i in roads:
            vertex = self.trees.vertices[i[0]]
            vertex.edges.append(Edge(self.trees.vertices[i[0]], self.trees.vertices[i[1]],i[2]))
        

    
    def escape(self, start, exits):
        """
        This function finds out the fastest route and the time taken from start to exits, after clawing one solulu tree
        I have 2 graphs on my escape function. One is the original graph where made based on all the roads
        and solulu tree. Another graph is the multi_graph, i made it according to the original graph but i change the roads to
        make sure it destroy the solulu tree before going to the exits. 

        Precondition: start needs to be a valid id tree in TreeMap, exits is a none empty list that contains all the 
                      exit tree IDs
                      
        Postcondition: return the shortest path from start to one of the exit tree after clawing one solulu tree
                       and return the time taken of it.

        Input:
            start: integer represent starting tree ID
            exits: list represent all the exits tree ID

        Return:
            (total_time, route):
            total_time: The total time taken to escape from start to exit
            route: list of route that player taken to exit

        Time complexity: 
            Best case analysis: O(5T + R + 2(R log T)) = O(T + R + R log T) = O(|R| log |T|) where R is the number of roads
                                in TreeMap and T is a set of unique tree in treemap

            Worst case analysis: O(|R| log |T|), first creating a multi graph is O(|T|) where T is the unique set of tree as we creating a set of unique tree graph. 
                                 connecting all the exits to the dummy ending vertex, the maximum exit can be the set of unique tree
                                 which is O(|T|) where T is the unique set of tree. We need to mark the solulu tree so we iterate through all the solulu tree. The worst case is 
                                 O(|T|) where T is the unique set of tree, as the set of unique tree can all be solulu tree. Running dijsktra on the tree graph is O(|R| log |T|)
                                 where R is set of roads and T is unique set of tree. Connect the starting vertex to all the solulu tree and maximum number of solulu tree can be the set of unique tree
                                 so it O(|T|) where T is the unique set of tree. Append roads to my new graph will be O(|R|) where R is the set of roads as we need to iterate through all the roads
                                 Run dijsktra on my new graph is O(|R| log |T|) where R is a set of roads and T is the unique set of tree. Backtracking is O(|T|) where T is the set of unique tree.
                                 Thus, the overall worst case complexity is O(|R| log |T|)

        Space complexity: 
            Input space analysis: O(|T|) where T is the set of exit tree as maximum exit tree can be all trees
            Aux space analysis: O(4T + R + 2(T)) = O(|T| + |R|) where T is the set of trees and R is the set of roads 
                                Creating a new graph which is O(|T|) space comp
                                for each exits we to connect exit to dummy ending vertex as exit can be T where T is the set of unique tree in input list so is O(|T|)
                                for each solulu tree we need to connect starting dummy vertex to solulu tree so is T where T is the set of unique tree in input list so is O(|T|)
                                running dijkstra on original graph is O(|T|) where T is the set of unique tree in input list
                                marking all the solulu tree is O(|T|) where T is the set of unique tree in input list
                                appending roads to all tree's vertex is O(|R|) where R is the set of roads in input list
                                running dijkstra on my new graph is O(|T|) where T is the set of unique tree in input list
                                backtracking is O(|T|) where T is the set of unique tree in input list
                                so overall is O(|T| + |R|)
        """
        source = self.trees.vertices[start] #setting source as start
        dummy_ending_vertex = Vertex(len(self.trees.vertices)) #insert dummy value
        dummy_starting_vertex = Vertex(len(self.trees.vertices) + 1) #starting dummy
        dummy_starting_vertex.id = start #setting starting dummy vertex id same as the start id

        #O(T) time comp where T is the number trees in TreeMap
        #O(T) space comp where T is the number of trees in TreeMap
        multi_graph = Graph(len(self.trees.vertices)) #create another graph
        multi_graph.vertices.append(dummy_ending_vertex)
        multi_graph.vertices.append(dummy_starting_vertex)  

        #O(T) time comp where T is the number of exits maximum X is T where T is the number of tree in graph
        #O(T) space comp where T is the number of exits 
        # each exit has edge connect to dummy value so overall is O(T)
        for i in exits:
            vertex = multi_graph.vertices[i] #connect all exits to dummy_ending_vertex
            vertex.edges.append(Edge(vertex, dummy_ending_vertex, 0))
        
        #O(1) aux space
        #O(T) time comp where T is the number of solulu trees maximum of solulu tree will be T where T 
        # number trees in TreeMap so overall O(T)
        self.mark_solulu(multi_graph, self.solulu_tree) #mark all the solulu tree in this graph
        source.distance = 0

        #O(R log T) time comp where R is the number of roads and T is the number of trees
        #O(T) space comp where T is the number of trees
        self.dijkstra(source, self.trees.vertices) #call dijsktra on original graph


        #O(T) time comp where T is the number of solulu trees as maximum solulu tree is T
        #O(T) space comp where T is the number of solulu trees
        self.connect_with_solulu_tree(multi_graph, self.solulu_tree)
    
        #O(R) time comp where R is the number of roads
        #O(R) space comp where R is the number of roads
        #let all new graph tree have the same old road
        for i in self.roads:
            vertex = multi_graph.vertices[i[0]]
            vertex.edges.append(Edge(multi_graph.vertices[i[0]], multi_graph.vertices[i[1]],i[2]))
        

        multi_graph.vertices[len(multi_graph.vertices) - 1].distance = 0
        #O(R log T) time comp where R is the number of roads and T is the number of trees
        #O(T) space comp where T is the number of trees
        self.dijkstra(multi_graph.vertices[len(multi_graph.vertices) - 1], multi_graph.vertices)
        last_vertex = multi_graph.vertices[len(self.trees.vertices)] #last vertex for back tracking
        
        total_time = last_vertex.distance
        route = []

        #O(T) time comp where T is the number of trees
        #O(T) space comp where T is the number of trees
        route = self.backtracking(start, multi_graph, route, last_vertex)
        route.reverse()
        return (total_time, route)
    
    def mark_solulu(self, graph, solulu):
        """
        mark all the solulu trees 
        iterate through all the solulu to mark the vertex

        precondition: a valid graph object and a list of solulus tree 
        postcondition: mark all the solulu trees in the vertex

        Input:
            graph: a valid graph object
            solulu: list of solulu trees with tuples
        Return:
            None

        Time complexity:
            Best case analysis: O(|T|) where T is a set of solulu trees maximum of solulu Trees will be the number of tree
                                in the graph

            Worst case analysis: Same as Best case
        
        Space complexity:
            Input space analysis: O(|T|) where T is a set of solulu trees, worst will be the number of solulu trees
            Aux space analysis: O(1) as it involved changing properties so no aux space

        """
        #O(T) where T is the number of solulu trees maximum of soluluTrees will be the number of tree
        #in the graph
        for i in solulu: #mark down all the solulu trees
            vertex = graph.vertices[i[0]]
            vertex.solulu = True
            vertex.teleport = graph.vertices[i[2]]
            vertex.teleport.solulu_from = vertex
            vertex.teleport.solulu_to = True #to determine this is the one teleport to
            vertex.time_claw = i[1]

    def connect_with_solulu_tree(self, graph, solulu):
        """
        connect the dummy starting point with all the solulu tree
        iterate all the solulu and add an edge from dummy starting point to the solulu tree

        precondition: a valid graph object and a list of solulus tree 
        postcondition: connect the dummy starting point with all the solulu tree

        Input:
            graph: graph that using 
            solulu: list of solulu trees to connect
        Return:
            None

        Time complexity:
            Best case analysis: O(|T|) where T is the set of solulu trees maximum of solulu Trees will be the number of tree
                                in the graph

            Worst case analysis: Same as Best case
        
        Space complexity:
            Input space analysis: O(|T|) where T is a set of solulu trees
            Aux space analysis: O(|T|) as appending all the roads to solulu trees into the starting vertex

        """
        #O(T) where T is the number of solulu tree
        #connect the new graph with solulu tree teleportation
        for i in solulu:
            u = graph.vertices[len(self.trees.vertices) + 1]
            v = graph.vertices[i[0]]            
            w = self.trees.vertices[i[0]].distance + graph.vertices[i[0]].time_claw
            v.teleport.previous = v
            v.previous = self.trees.vertices[i[0]].previous
            u.edges.append(Edge(u,v.teleport,w)) #connect the dummy starting point to all solulu trees
    
    def backtracking(self, start, graph, route, last_vertex):
        """
        for dijsktra to back track from the exit to the start
        using its properties .previous to back track one by one
        first start with new graph, if it meets a solulu tree then 
        start with the solulu tree in the original graph and continue backtrack to the start

        precondition: start is a valid id of the starting vertex. 
                      graph is a valid graph object that representing the treemap
                      route is a empty list for appending the route from start to exit
                      last_vertex is a valid vertex object representing the ending vertex

        postcondition: return the route from starting to the exit

        Input:
            start is a valid id of the starting vertex. 
            graph is a valid graph object that representing the treemap
            route is a empty list for appending the route from start to exit
            last_vertex is a valid vertex object representing the ending vertex

        Return:
            route: the route from starting to ending point

        Time complexity:
            Best case analysis: O(2* |T|) = O(|T|) where T is a set of tree
            Worst case analysis: O(|T|) where T is a set of tree. When it needs to visit
                                 every single vertex in the graph. So is O(|T|)
        
        Space complexity:
            Input space analysis: O(1)
            Aux space analysis: O(|T|) where T is a set of tree as route needs at least T space

        """
        condition1 = last_vertex.previous.solulu_from != None #if the vertex is teleported from a solulu
        condition2 = last_vertex.previous.id == graph.vertices[len(self.trees.vertices)].previous.id #if last_vertex is the ending vertex
        condition3 = last_vertex.previous != last_vertex.previous.solulu_from #if solulu is not itself 
        
        #O(T) the number of trees in the graph
        for i in range(len(graph.vertices)):
            if condition1 and condition2 and condition3:       #if ending vertex is teleported from a solulu
                route.append(last_vertex.previous.id)          #append the ending vertex 
                route.append(last_vertex.previous.solulu_from.id)    #then append the solulu tree
                current = self.trees.vertices[last_vertex.previous.solulu_from.id] #set current to the solulu tree
                break
            
            elif last_vertex.previous != graph.vertices[len(graph.vertices) - 1]: #if the vertex previous is not starting point
                last_vertex = last_vertex.previous                          #continue until it reach the vertex before starting point
                route.append(last_vertex.id)                                #vertex before starting point is a solulu tree as we travel to solulu first
            
            else:
                current = self.trees.vertices[last_vertex.id]        #if the vertex previous is starting point means we reach solulu tree
        
        #O(T) the number of trees in the graph
        for i in range(len(graph.vertices)):          #start with solulu tree
            if current != self.trees.vertices[start]: #check whether reached the start
                current = current.previous            #backtrack how to get to solulu tree
                route.append(current.id)              #if havent reach start continue append
        
        return route
    
    def dijkstra(self, source, v):
        """
        Thie function is to obtain every shortest distance from the source to the vertex
        using min heap as the array and keep serving until all vertex has been served

        Precondition: source is a valid vertex ID that representing starting point
                      v is a set of vertex that dijkstra is using to obtain shortest distance from source
                      to each vertex

        Postcondition: we have the shortest distance from source to all vertex

        Input:
            source: vertex ID representing starting vertex
            v: a list representing tree vertices
        Return:
            None

        Time complexity: 
            Best case analysis: O(|R| log |T|) where R is the edges of all trees and T is the number of trees 

            Worst case analysis: O(|R| log |T|), as we serving all the trees so is O(|T|) where T is the number of trees in v
                                For each tree we need to iterate through its edges so it is O(T^2) as we need to iterate through
                                all the trees and each tree can have up to T-1 edges so is O(|T|^2) where T is the number of trees
                                For every edges we update the distance so is O(|T|^2 log |T|). Where T is the number of trees. 
                                
                                T * (log T + T log T) = T log T + T^2 log T = O(T^2 log T) so overall the complexity is O(|T^2| log |T|)
                                but when the graph is dense T^2 = R so is O(|R| log |T|) where R is the edges of all trees.

        Space complexity: 
            Input space analysis: O(|T|) where T is the number of tree vertices, source is a vertex id where v is the set of tree vertices
            Aux space analysis: O(2|T|) = O(|T|) where T is the number of tree vertices as initialising index_mapping array space is O(|T|) where T is the set of tree vertices 
                                and minheap array is space O(|T|) where T is the number of trees. So overall is O(|T|)
        """
        discovered = MinHeap(len(v)) #create discovered as minHeap 
                                     #O(T) space comp where T is the number of tree vertices
        index_mapping = [None] * len(v) #O(T) space comp where T is the number of tree vertices
        discovered.add(source, index_mapping) #add source to heap first

        #O(T) time comp where T is the number of trees
        while len(discovered) > 0:
            #O(log T) time comp where T is the number of trees
            (index_mapping, u) = discovered.serve(index_mapping) 
            u = discovered.the_array[index_mapping[u.id]]
            u.visited = True
            
            #O(T) time comp where T is the number of tree  
            #as one tree has maximum of T - 1 outgoing edges so is T
            for edge in u.edges:
                v = edge.v
                if v.discovered == False: #means distance still inf
                    v.discovered = True
                    v.distance = u.distance + edge.w
                    v.previous = u
                    discovered.add(v, index_mapping)
                    
                elif v.visited == False:
                    #if i find a shorter one, change it
                    if v.distance > u.distance + edge.w:
                        v.distance = u.distance + edge.w
                        v.previous = u
                        #O(log T) time comp where T is the number of trees
                        discovered.update(v, index_mapping)

    
class MinHeap():
    """
    a data structure, heap where the smallest element will be on the root
    Note: This data structure is retrieved from the prerequisite unit FIT1008
        it is used to modify so it can be use with dijkstra
    """
    def __init__(self, size) -> None:
        """
        initialise the attribute of min heap based on size given

        precondition: size is an integer and non negative
        postcondition: created a minHeap object with the size

        Input:
            size: integer representing the size of graph 
        Return:
            None

        Time complexity:
            Best case analysis: O(1)
            Worst case analysis: Same as Best case as it need to go through no matter what
        
        Space complexity:
            Input space analysis: O(1)
            Aux space analysis: O(N) where N is the size of heap

        """
        self.length = 0
        self.the_array = [None] * (size + 1) 

    def __len__(self) -> int:
        """
        return the length of the minheap

        precondition: None 
        postcondition: return length of minheap

        Input:
            None
        Return:
            integer representing length of the heap

        Time complexity:
            Best case analysis: O(1)
            Worst case analysis: Same as Best case as it need to go through no matter what
        
        Space complexity:
            Input space analysis: O(1)
            Aux space analysis: O(1)

        """
        return self.length

    def is_full(self) -> bool:
        """
        return boolean representing min heap is full/ not full

        precondition: None
        postcondition: return boolean representing full/not full

        Input:
            None
        Return:
            boolean representing if is full

        Time complexity:
            Best case analysis: O(1)
            Worst case analysis: Same as Best case as it need to go through no matter what
        
        Space complexity:
            Input space analysis: O(1)
            Aux space analysis: O(1)

        """
        return self.length == len(self.the_array)

    def rise(self, k: int, index_mapping) -> None:
        """
        rise if it is smaller than its parent in heap

        precondition: k is a valid index in the minheap array
                      index_mapping is a list that indicates the index of the minheap

        postcondition: The smaller child swap position with its parent, if the parent larger than its child

        Input:
            k: integer representing the item that rising
            index_mapping: list that representing index of the vertices

        Return:
            None

        Time complexity:
            Best case analysis: O(1) if k element is larger than its parent
            Worst case analysis: O(log N) where N is the size of heap as we need to rise to top
                                 when k element is the smallest
        
        Space complexity:
            Input space analysis: O(N) where N is the size of heap
            Aux space analysis: O(1) as we just swapping item
        """
        item = self.the_array[k]

        #if its parent is larger than its child then swap
        while k > 1 and item.distance < self.the_array[k // 2].distance:
            self.the_array[k] = self.the_array[k // 2]
            index_mapping[self.the_array[k].id] = k
             
            k = k // 2
        self.the_array[k] = item 
        index_mapping[item.id] = k 


    def add(self, element, index_mapping) -> bool:
        """
        add a new element to the heap and rise if is needed

        precondition: element is a valid id 
                      index_mapping is a list that keep track index of the heap

        postcondition: the element is added inside the heap and it is in the correct position 

        Input:
            element: vertex representing element to add to heap
            index_mapping: list representing index of the heap
        Return:
            None

        Time complexity:
            Best case analysis: O(1) if element is the largest so no need rise
            Worst case analysis: O(log N) where N is the number of element in heap when element is smallest value where N is the size of heap as we need to rise to top
        
        Space complexity:
            Input space analysis: O(N) where N is the size of heap 
            Aux space analysis: O(1) as we just swapping item
        """
        if self.is_full():
            raise IndexError
        
        self.length += 1
        self.the_array[self.length] = element
        index_mapping[element.id] = self.length
        #add to the end of heap then rise
        self.rise(self.length, index_mapping)
        
        

    def smallest_child(self, k: int) -> int:
        """
        get the smallest child between its parent and child

        precondition: k is a valid index in the min heap
        postcondition: return the smallest child index 

        Input:
            k: integer representing the item that rising
        Return:
            integer representing index of element in heap

        Time complexity:
            Best case analysis: O(1) as we just comparing
            Worst case analysis: same as best case
        
        Space complexity:
            Input space analysis: O(1) 
            Aux space analysis: O(1) as we just compare and returning index
        """
        #if its parent is larger than its child then return its child
        if 2 * k == self.length or \
                self.the_array[2 * k].distance < self.the_array[2 * k + 1].distance:
            return 2 * k
        else:
            return 2 * k + 1

    def sink(self, k: int, index_mapping) -> None:
        """
        swap with its child if is larger than its child in heap

         precondition: k is a valid index in the minheap array
                       index_mapping is a list that indicates the index of the minheap
        
         postcondition: return the heap with all the element in correct position

        Input:
            k: integer representing the item that sinking
            index_mapping: list that representing index of the vertices
        Return:
            index_mapping: list that representing index of the vertices

        Time complexity:
            Best case analysis: O(1) if k index element is smaller than its child
            Worst case analysis: O(log N) where N is the number of element in heap when k index element is the largest and need to sink to bottom, where N is the size of heap as we need to sink to bottom
        
        Space complexity:
            Input space analysis: O(N) where N is the size of heap
            Aux space analysis: O(1) as we just swapping item
        """
        item = self.the_array[k]
        
        #loop through all parent to check if is smaller than its child
        while 2 * k <= self.length:
            min_child = self.smallest_child(k)
            if self.the_array[min_child].distance >= item.distance:     #swap its child and parent if is smaller
                break
            index_mapping[self.the_array[k].id] = k #update the index mapping
            self.the_array[k] = self.the_array[min_child] 
            k = min_child
    
        self.the_array[k] = item
        index_mapping[item.id] = min_child
        return index_mapping

    def get_min(self, index_mapped):
        """ 
        Remove (and return) the min element from the heap. 

        precondition: index_mapping is a list that indicates the index of the minheap
        postcondition: return the minimum element in the heap which is the root

        Input:
            index_mapping: list that representing index of the vertices
        Return:
            min_elt representing minimum vertex in the heap

        Time complexity:
            Best case analysis: O(log N) where N is the number of element in heap as we need to swap the largest item to the top and sink it no matter what
            Worst case analysis: same as best case
        
        Space complexity:
            Input space analysis: O(N) where N is the size of heap 
            Aux space analysis: O(1) as we just swapping item
        """
        
        if self.length == 0:
            raise IndexError

        min_elt = self.the_array[0]
        self.length -= 1
        if self.length > 0:
            self.the_array[0] = self.the_array[self.length]
            self.sink(1, index_mapped)
        return min_elt
    
    def update(self, vertex, index_mapping):
        """ 
        update the vertex position according to its value and rise

        precondition: vertex is a valid vertex to update
        postcondition: return the index mapping representing the heap and it is update in correct position

        Input:
            vertex: vertex to update in heap
                    index_mapping is a list that indicates the index of the minheap
            index_mapping: list that representing index of the vertices

        Return:
            index_mapping: list that representing index of the vertices

        Time complexity:
            Best case analysis: O(1) when no need to rise if it is the largest element
            Worst case analysis: O(log N) where N is the number of element in heap when it is smallest element then we need rise to top

        Space complexity:
            Input space analysis: O(N) where N is the size of heap 
            Aux space analysis: O(1) as we just swapping item
        """
        index_mapping = self.rise(index_mapping[vertex.id], index_mapping)
        return index_mapping

    def serve(self, index_mapping):
        """ 
        remove the root and return the vertex

        precondition: index_mapping is a list that indicates the index of the minheap
        postcondition: the root is served and returned, the next root will be the smallest element 
                       in the heap 

        Input:
            index_mapping: list that representing index of the vertices

        Return:
            index_mapping: list that representing index of the vertices
            item: vertex representing the root and the smallest element

        Time complexity:
            Best case analysis: O(log N) where N is the number of element in heap as we need to sink the root 
                                after swapping the served element and the last element. To make sure it is in correct position
            Worst case analysis: same as best case

        Space complexity:
            Input space analysis: O(N) where N is the size of heap 
            Aux space analysis: O(1) as it is just swapping
        """

        #swapping root with the last element
        self.length -= 1
        item = self.the_array[1]
        self.the_array[1] = self.the_array[self.length + 1]
        self.the_array[self.length + 1] = item
        index_mapping[item.id] = self.length + 1
        index_mapping[self.the_array[1].id] = 1

        #sink the last element from root to make sure it is in correct position
        if self.length > 1:
            index_mapping = self.sink(1, index_mapping)

        
        return (index_mapping , item)
    
    def __str__(self) -> str:
        """ 
        magic function for heap to print

        precondition: there is heap created
        postcondition: print the string representing heap

        Input:
            None
        Return:
            return_string: a string represent heap

        Time complexity:
            Best case analysis: O(N) where N is the number of element in heap
            Worst case analysis: same as best case

        Space complexity:
            Input space analysis: O(1)
            Aux space analysis: O(1)
        """
        return_string = ""
        for i in range(1, self.length + 1):
            return_string = return_string + str(self.the_array[i]) + ","
        return return_string
        

class Graph:
    """
    To create a graph with vertex
    """
    def __init__(self, v):
        """ 
        magic function to initialise vertices

        precondition: v is an integer and non negative value
        postcondition: a graph object representing the graph is created

        Input:
            v: integer representing number of vertex in the graph

        Return:
            None

        Time complexity:
            Best case analysis: O(N) where N is the size of graph
            Worst case analysis: same as best case

        Space complexity:
            Input space analysis: O(1)
            Aux space analysis: O(N) where N is the size of graph
        """
        self.vertices = [None] * v
        for i in range(v):
            self.vertices[i] = Vertex(i)

    def __str__(self):
        """ 
        magic function for graph to print

        precondition: there is a graph object created
        postcondition: print the string representing graph 

        Input:
            None
        Return:
            return_string: a string represent graph

        Time complexity:
            Best case analysis: O(N) where N is the number of vertices in graph
            Worst case analysis: same as best case

        Space complexity:
            Input space analysis: O(1)
            Aux space analysis: O(1)
        """
        return_string = ""
        for vertex in self.vertices:
            return_string = return_string + str(vertex) + ","
        return return_string 
    
class Vertex:
    """
    To create a vertex with this class
    """
    def __init__(self, id) -> None:
        """ 
        magic function to initialise value of vertex

        precondition: id is an integer
        postcondition: a vertex is created with the id

        Input:
            id representing id of vertex

        Return:
            None

        Time complexity:
            Best case analysis: O(1) as it is initialisation
            Worst case analysis: O(1) as it is initialisation

        Space complexity:
            Input space analysis: O(1)as it is initialisation
            Aux space analysis: O(1) as it is initialisation

        """
        self.id = id
        self.edges = []
        self.discovered = False
        self.visited = False
        self.distance = float('inf')
        self.previous = None
        self.solulu_from = None
        self.solulu_to = False
        self.teleport = None
        self.time_claw = 0


    def __str__(self):
        """ 
        magic function for vertex to print

        precondition: a vertex object is created
        postcondition: print a string representing vertex 

        Input:
            None
        Return:
            return_string: a string represent vertex

        Time complexity:
            Best case analysis: O(1) as it is just initialisation
            Worst case analysis: O(1) as it is just initialisation

        Space complexity:
            Input space analysis: O(1)as it is just initialisation
            Aux space analysis: O(1)as it is just initialisation
        """
        return_string = str(self.id)
        return return_string

class Edge:
    """
    class to create edge
    """
    def __init__(self, u, v ,w):
        """ 
        magic function to initialise edge

        precondition: u and v is a valid integer representing trees, w is a valid integer representing weight
        postcondition: an edge object is created 

        Input:
            None
        Return:
            None

        Time complexity:
            Best case analysis: O(1) as it is just initialisation
            Worst case analysis: O(1)as it is just initialisation

        Space complexity:
            Input space analysis: O(1)as it is just initialisation
            Aux space analysis: O(1)as it is just initialisation
        """
        self.u = u
        self.v = v
        self.w = w

