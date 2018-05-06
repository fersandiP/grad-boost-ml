import numpy as np

class DecisionTree:
    def __init__(self, max_depth=2, min_size=None, func='gini'):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None
        self.func = func
        
    def fit(self, dataset, label):
        new_dataset = dataset.copy()
        new_dataset['label'] = label
        
        self.dataset = new_dataset.as_matrix()
        self.label = list(set(label))
        
        if self.min_size is None:
            self.min_size = len(self.dataset)/10
            
        self.root = self._split_tree(self.dataset)
        self._split(self.root, 1)
    
    def predict(self, dataset):
        if self.root is None:
            raise "Decison Tree belum di fit"
            
        rows = dataset.as_matrix()
        
        return np.asarray([self._predict(self.root, row) for row in rows])
            
                
    def _predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']

    def evaluate(self, test_data):
        pass
    
    def _calculate_gini_index(self, groups):
        instances = sum(len(group) for group in groups)
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            for class_val in self.label:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / instances)
        return gini
    
    def _calculate_sse_func(self, groups):
        sse = 0.0
        for group in groups:
            if len(group) == 0:
                continue
            y_mean = sum(row[-1] for row in group)/len(group)
            sse += sum((row[-1] - y_mean)**2 for row in group)
        return sse
    
    def _calculate_cost(self, groups):
        if self.func == 'gini':
            return self._calculate_gini_index(groups)
        elif self.func == 'sse':
            return self._calculate_sse_func(groups)
        raise 'Unknown function'
    
    def _split_tree(self, dataset):
        b_index, b_value, b_score, b_groups = 999, 999, None, None
        for index in range(len(dataset[0])-1):

            col_data = set([data[index] for data in dataset])
                
            for col in col_data:
                groups = self._test_split(index, col, dataset)
                cost = self._calculate_cost(groups)
                if b_score is None or cost < b_score:
                    b_index, b_value, b_score, b_groups = index, col, cost, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}
    
    def _test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def _split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self._to_terminal(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self._to_terminal(left)
        else:
            node['left'] = self._split_tree(left)
            self._split(node['left'], depth+1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self._split_tree(right)
            self._split(node['right'], depth+1)
    
    def _to_terminal(self, group):
        if self.func == 'gini':
            return self._to_terminal_gini(group)
        elif self.func == 'sse':
            return self._to_terminal_regression(group)
    
    def _to_terminal_gini(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
    
    def _to_terminal_regression(self, group):
        return sum(row[-1] for row in group)/len(group)
