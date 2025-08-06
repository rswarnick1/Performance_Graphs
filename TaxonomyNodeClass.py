import numpy as np
import scipy.stats as stats
import scipy as sp
import matplotlib.pyplot as plt

# https:// anytree.readthedocs.io/en/latest/api.html
from anytree import Node, RenderTree


class TaxonomyNode(Node):

    """ A class representing a node in a taxonomy tree.
    Each node can have children, a distribution for D given C, a distribution for C, and a threshold eta.
    The node can also calculate precision and recall based on these distributions.
    Attributes: 

        name (str): The name of the node.
        distribution_D_C (list): A list of two distributions, one for each class (0 and 1).
        distribution_C (scipy.stats.rv_continuous): A Bernoulli distribution with some given probability.
        eta (float): The threshold for the distribution_D_C[0] or distribution_D_C[1].
        parent (TaxonomyNode): The parent node in the taxonomy tree.
        children (list): A list of child nodes in the taxonomy tree.
        direction (str): The direction of the outlier ('greaterthan' or 'lessthan').
        precision (float): The precision calculated for the node.
        recall (float): The recall calculated for the node.
    Methods:
        set_distribution_D_C(distribution_D_C): Sets the distribution_D_C for the node.
        set_distribution_C(distribution_C): Sets the distribution_C for the node.
        prec_recall_calc(direction): Calculates precision and recall based on the distributions and eta.
        print_tree(): Prints the tree structure starting
        from the current node.
        add_child(child_node): Adds a child node to the current node.
    """

    def __init__(self, name, distribution_D_C = None, distribution_C = None, eta = None, parent = None, children = None, direction = "greaterthan"):
        super().__init__(name, parent=parent, children=children)
        self.name = name
        # list of the children taxonomynode objects
        self.children = []
        # distribution_D_C is a list of two distributions, one for each class (0 and 1). These must have the same support.
        # distribution_C is a bernoulli with some given probability (dictating which of the elements of the distribution_D_C list it appropriate)
        self.distribution_D_C = distribution_D_C
        self.distribution_C = distribution_C
        # If precision and recall are calculated, they will be stored in these attributes
        self.precision = None
        self.recall = None
        # eta is the threshold for the distribution_D_C[0] or distribution_D_C[1]
        if children is None:
            self.children = []
        else:
            self.children = children
        # eta is the threshold for the distribution_D_C[0] or distribution_D_C[1]
        self.eta = eta
        # direction is the direction of the outlier (scoring less than or greater than the specified threshold, for P(D=P) in precision and recall)
        self.direction = direction
        #self.height = height

    def set_distribution_D_C(self, distribution_D_C = None):
        if not self.children:
            # If no children exist, set the distribution_D_C directly
            if distribution_D_C is None:
                raise ValueError("distribution_D_C must be provided for leaf nodes.")
            self.distribution_D_C[0] = distribution_D_C[0]
            self.distribution_D_C[1] = distribution_D_C[1]
        else:
            # If children exist, calculate the distribution_D_C for the current node based on the children's distributions
            self.distribution_D_C = [None, None]

            loc = [0]*2
            scale = [1]*2

            loc[0]= sum([child.distribution_D_C[0].mean() for child in self.children])
            scale[0] = np.sqrt(sum([child.distribution_D_C[0].std()**2 for child in self.children]))
            self.distribution_D_C[0] = stats.norm(loc=loc[0]*(1-self.distribution_C.mean()), scale=scale[0])

            loc[1]= sum([child.distribution_D_C[1].mean()*self.distribution_C.mean() for child in self.children])
            scale[1] = np.sqrt(sum([child.distribution_D_C[1].std()**2 for child in self.children]))
            self.distribution_D_C[1] = stats.norm(loc=loc[1], scale=scale[1])


    # Method to set distribution of indicator C for current node if node is a leaf node
    def set_distribution_C(self, distribution_C = None):
        if not self.children:
            self.distribution_C = distribution_C
        else:
            self.distribution_C = stats.bernoulli(1-np.prod(np.array([1-child.distribution_C.mean() for child in self.children])))
            

    # Method to put eta in for current node as threshold for distribution_D_C[0] or distribution_D_C[1]
    # Method to calculate precision and recall for current node
    def prec_recall_calc(self, direction = "left"):
        if self.distribution_D_C is not None and self.distribution_C is not None:
            if self.direction == "greaterthan":
                # Calculate precision and recall using the distributions
                self.precision = self.distribution_D_C[1].sf(self.eta) * self.distribution_C.mean()/(self.distribution_D_C[1].sf(self.eta)*(1-self.distribution_C.mean()) + self.distribution_D_C[1].cdf(self.eta)*self.distribution_C.mean())
                self.recall = self.distribution_D_C[1].sf(self.eta)
            elif self.direction == "lessthan":
                self.precision = self.distribution_D_C[1].cdf(self.eta) * self.distribution_C.mean()/(self.distribution_D_C[1].cdf(self.eta)*(1-self.distribution_C.mean())+ self.distribution_D_C[1].cdf(self.eta)*self.distribution_C.mean())
                self.recall = self.distribution_D_C[1].cdf(self.eta)                   
            else:
                print("Invalid direction specified. Use 'greaterthan' or 'lessthan'.")
                return
        else:
            print("Distributions not set for current TaxonomyNode.")
            return
    def print_tree(self):
        for pre, _, node in RenderTree(self):
            print(f"{pre}{node.name}(precision:{node.precision},recall:{node.recall},eta:{node.eta},distribution_C:{node.distribution_C}, distribution_D_C: {node.distribution_D_C})")

    # Method to add a child node to current iteration of node class
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    def __repr__(self):
        return f"TaxonomyNode({self.name})"


if __name__ == "__main__":
    # Example usage
    root = TaxonomyNode("Root")
    child1 = TaxonomyNode("Child1", parent=root)
    child2 = TaxonomyNode("Child2", parent=root)
    root.add_child(child1)
    root.add_child(child2)
    print(RenderTree(root))