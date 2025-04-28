#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <queue>
#include <climits>
#include <algorithm>
#include <fstream>

class DataPoint{
public:
    DataPoint() { }
    DataPoint(const std::vector<int> features, const int label) : label(label), features(features) { }

    int label = 0;
    std::vector<int> features;
};

class Node {
public:
  Node() {}
  virtual ~Node() {}
  virtual bool is_leaf() const = 0;
};

class RuleNode : public Node {
public:
  RuleNode(int split_feature) : split_feature(split_feature) {}
  ~RuleNode() {
    if (left) delete left;
    if (right) delete right; }

  bool is_leaf() const override { return false; }
  
  int split_feature = 0;
  Node *left = nullptr;
  Node *right = nullptr;
};



class LeafNode : public Node {
public:
  LeafNode() {}
  LeafNode(std::vector<DataPoint>& data, double const entropy, int const n_classes);
  ~LeafNode() { }

  bool is_leaf() const override { return true; }
  
  void set_data(std::vector<DataPoint>& data) { this->data = data; }
  void set_prediction(int prediction) { this->prediction = prediction; }
  void set_info_gain(double info_gain) { this->info_gain = info_gain; }
  void set_entropy(double entropy) { this->entropy = entropy; }
  void set_n_classes(unsigned n_classes) { this->n_classes = n_classes; }

  std::vector<DataPoint> get_data() const { return data; }
  int get_prediction() const { return prediction; }
  double get_entropy() const { return entropy; }
  double get_info_gain() const { return info_gain; }
  unsigned get_n_samples() const { return n_samples; }
  std::unordered_map<int, unsigned> get_class_proportion() const { return class_proportion; }
  
private:
  int prediction = 0;
  double entropy = 0.5;
  double info_gain = 0;
  unsigned n_classes = 0;
  unsigned n_samples = 0;
  
  std::unordered_map<int, unsigned> class_proportion; // first variable in map can be changed to any type depending on the type of the names of classes in data
  std::vector<DataPoint> data;
  
  void calculate_class_proportion();
  void set_prediction();
};

LeafNode::LeafNode(std::vector<DataPoint>& data, double const entropy, int const n_classes) : data(data), entropy(entropy), n_classes(n_classes), n_samples(data.size()) {

  this->calculate_class_proportion();
  this->set_prediction();
}

void LeafNode::calculate_class_proportion()
{
  for (DataPoint& datapoint : data)
    class_proportion[datapoint.label]++;

}

void LeafNode::set_prediction()
{
   auto max = std::max_element(class_proportion.begin(), class_proportion.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second <= b.second;
            });

   this->prediction = max->first;
}


struct CompareLeaf {
  // big information gain is prioritised
  bool operator()(const LeafNode *l1, const LeafNode *l2) {
    return l1->get_info_gain() < l2->get_info_gain();
  }
};
							     

class DecisionTree {
public:
  DecisionTree(int const max_leaf_nodes = INT_MAX, unsigned min_samples_split = 5, double min_impurity_decrease = 0.05, double min_impurity = 0.01);
  ~DecisionTree() { if (root) delete root; }

  void fit(std::vector<std::vector<int>>& features, std::vector<int>& labels);
  
  std::vector<int> predict(std::vector<std::vector<int>>& features); // returns the predicted classes for given features
  int predict(std::vector<int>& features); // returns single class for given features for one datapoint
  
private:
  Node *root = nullptr;
  unsigned n_classes = 0;
  int max_leaf_nodes = INT_MAX;
  unsigned min_samples_split = 5;
  double min_impurity_decrease = 0.05;
  double min_impurity = 0.01;
  
  void set_n_classes(std::vector<int>& labels); // calculates the number of classes in data
  
  void build_tree(std::vector<DataPoint>& data);
  double calculate_entropy(std::vector<DataPoint>& data);
  std::pair<std::pair<std::vector<DataPoint>, std::vector<DataPoint>>, std::pair<int, double>> split(std::vector<DataPoint>&);
  std::pair<int, double> find_best_split(std::vector<DataPoint>& data);
};

DecisionTree::DecisionTree(int const max_leaf_nodes, unsigned min_samples_split, double min_impurity_decrease, double min_impurity) : max_leaf_nodes(max_leaf_nodes), min_samples_split(min_samples_split), min_impurity_decrease(min_impurity_decrease), min_impurity(min_impurity) {}

void DecisionTree::fit(std::vector<std::vector<int>>& features, std::vector<int>& labels)
{
  std::vector<DataPoint> data(features.size());

  for (int i = 0; i < features.size(); i++)
      data[i] = DataPoint(features[i], labels[i]);
  
  this->set_n_classes(labels);
  this->build_tree(data);
}

void DecisionTree::set_n_classes(std::vector<int>& labels)
{
  std::set classes(labels.begin(), labels.end());
  this->n_classes = classes.size();
}

double DecisionTree::calculate_entropy(std::vector<DataPoint>& data)
{
  double squared_sum = 0;
  unsigned size = data.size();

  std::unordered_map<int, unsigned> class_proportion;

  for (DataPoint& dp : data)
    class_proportion[dp.label]++;

  for (auto it = class_proportion.begin(); it != class_proportion.end(); ++it) 
    squared_sum += (static_cast<double>(it->second) / size) * (static_cast<double>(it->second) / size);

  return (1 - squared_sum);
}


std::pair<int, double> DecisionTree::find_best_split(std::vector<DataPoint>& data)
{

  unsigned size = data.size();
  double best_entropy = INT_MAX;
  int best_split_index = -1;

  for (unsigned i = 0; i < data[0].features.size(); ++i) {
    // split data by index i. then find overall entropy, if it is less than best entropy, update the value of best_entropy
    
     std::vector<DataPoint> current_left;
     std::vector<DataPoint> current_right;
    for (DataPoint& dp : data) {
      if (dp.features[i] == 1)
	current_left.push_back(dp);
      else
	current_right.push_back(dp);
      
    }

    // overall entropy is the weighted sum of entropies
    double current_entropy = ((static_cast<double>(current_left.size()) / size) * calculate_entropy(current_left) + (static_cast<double>(current_right.size()) / size) * calculate_entropy(current_right));

    // update best_entropy if necessary
    if (current_entropy < best_entropy) {
      best_entropy = current_entropy;
      best_split_index = i;
      
    }
      
  }

  //std::cout << "entropy: " << best_entropy << '\n';
  
  return std::make_pair(best_split_index, best_entropy);
  
}


std::pair<std::pair<std::vector<DataPoint>, std::vector<DataPoint>>, std::pair<int, double>> DecisionTree::split(std::vector<DataPoint>& data)
{
  
  std::vector<DataPoint> left;
  std::vector<DataPoint> right;

  auto [best_split_index, best_entropy] = find_best_split(data);
  
  if (best_split_index == -1) { std::cout << "couldnt find best split\n";}

  for (DataPoint& dp : data) {
    if (dp.features[best_split_index] == 1)
      left.push_back(dp);
    else
      right.push_back(dp);
  }

  //std::cout << "split() function end\n";
  return std::make_pair(std::make_pair(left, right), std::make_pair(best_split_index, best_entropy));
  
}

void DecisionTree::build_tree(std::vector<DataPoint>& data)
{
  // prioritised by greater information gain values
  std::priority_queue<LeafNode*, std::vector<LeafNode*>, CompareLeaf> pq;

  std::unordered_map<Node*, std::pair<RuleNode*, bool>> node_to_parent; // bool = true for left child

  double entrp = this->calculate_entropy(data);
  this->root = new LeafNode(data, entrp, this->n_classes);
   
  pq.push(dynamic_cast<LeafNode*>(this->root));

  while (!pq.empty() && pq.size() < max_leaf_nodes) {
    LeafNode *current_leaf = pq.top();
    pq.pop();

    // check criterions
    if (current_leaf->get_n_samples() < min_samples_split || current_leaf->get_entropy() <= min_impurity)
      continue;

    // split the data
    std::vector<DataPoint> current_data= current_leaf->get_data();
    
    auto [split_data, split] = this->split(current_data);

    // unpack the results of split() function
    int split_feature = split.first;
    double split_entropy = split.second;

    // check criterions
    if (current_leaf->get_entropy() - split_entropy < min_impurity_decrease)
      continue;
    
    std::vector<DataPoint> left_data = split_data.first;
    std::vector<DataPoint> right_data = split_data.second;

    
    // entropies
    double left_entrp = this->calculate_entropy(left_data);
    double right_entrp = this->calculate_entropy(right_data);
    double current_entrp = current_leaf->get_entropy();

    // create child nodes
    LeafNode *left = new LeafNode(left_data, left_entrp, this->n_classes);
    LeafNode *right = new LeafNode(right_data, right_entrp, this->n_classes);

    // set information gains for child nodes
    left->set_info_gain((current_entrp - left_entrp));
    right->set_info_gain((current_entrp - right_entrp));

    // convert the current leaf node to rule node
    RuleNode *rule_node = new RuleNode(split_feature);
    rule_node->left = left;
    rule_node->right = right;

    // Track parent relationship for new nodes
    node_to_parent[left] = {rule_node, true};   // left child
    node_to_parent[right] = {rule_node, false}; // right child
    
    // Replace the current_leaf with rule_node in the tree
    if (this->root == current_leaf) {
      this->root = rule_node; 
    } else {
      // Find the parent and update the appropriate pointer
      auto parent_info = node_to_parent[current_leaf];
      RuleNode* parent = parent_info.first;
      bool is_left_child = parent_info.second;
      
      if (is_left_child) {
        parent->left = rule_node;
      } else {
        parent->right = rule_node;
      }
    }
    
    auto parent_it = node_to_parent.find(current_leaf);
    if (parent_it != node_to_parent.end()) {
      auto parent_info = parent_it->second;  //sSave the parent info
      node_to_parent.erase(parent_it);  // erase the old entry
      node_to_parent[rule_node] = parent_info;  // add the new relationship
    }
    
    // Now it's safe to delete the old leaf
    delete current_leaf;
    

    //std::cout << "left->info_gain: " << left->get_info_gain() << std::endl;
    //std::cout << "right->info_gain: " << right->get_info_gain() << std::endl;
    
    // push child nodes to priority queue
    pq.push(left);
    pq.push(right);
    
  }
  
}

std::vector<int> DecisionTree::predict(std::vector<std::vector<int>>& datapoints)
{

  std::vector<int> predictions(datapoints.size(), 0);

  unsigned i = 0;
  for (std::vector<int>& dp : datapoints) {
    Node *root = this->root;

    // unless leaf node traverse tree according to rules
    while (!(root->is_leaf())) {
      int split_f = dynamic_cast<RuleNode*>(root)->split_feature;

      if (dp[split_f] == 1) // go left
	root = dynamic_cast<RuleNode*>(root)->left;

      else if (dp[split_f] == 0) // go right
	root = dynamic_cast<RuleNode*>(root)->right;
    }

    predictions[i++] = dynamic_cast<LeafNode*>(root)->get_prediction();
  }

  return predictions;
  
}

int main()
{
    std::fstream training_file ("training.dat", std::ios::in);
    std::vector<std::vector<int>> X;
    std::vector<int> Y;
    
    // read data from training.dat
    while (!training_file.eof()) {
          std::string line;
          std::vector<int> row;
          int count = 0;

	  std::getline(training_file, line);
	  
          for (int i = 0; i < line.size(); i++) {
            if (count == 10 && (line[i] == '1' || line[i] == '0' || line[i] == '2')) Y.push_back(line[i] - '0');

            else {
                if (line[i] == '1') { row.push_back(1); count++; }
                else if (line[i] == '0') { row.push_back(0); count++; }
            }
          }

          if (row.size() > 0) X.push_back(row);

    }

    training_file.close();

    // read test.dat

    std::fstream test_file ("testing.dat", std::ios::in);
    std::vector<std::vector<int>> X_test;
    std::vector<int> Y_test;

    while (!test_file.eof()) {
      std::string line;
      std::vector<int> row;
      int count = 0;

      std::getline(test_file, line);
      
      for (int i = 0; i < line.size(); i++) {
          if (count == 10 && (line[i] == '1' || line[i] == '0' || line[i] == '2')) Y_test.push_back(line[i] - '0');

          else {
            if (line[i] == '1') { row.push_back(1); count++; }
            else if (line[i] == '0') { row.push_back(0); count++; }
          }
      }

       if (row.size() > 0) X_test.push_back(row);

    }

    test_file.close();

    
    // create model and fit the data
    DecisionTree tree = DecisionTree(INT_MAX, 5, 0.03, 0.1);
    tree.fit(X, Y);

    std::vector<int> Y_predict = tree.predict(X_test);

    // print predictions alongside test labels
    std::cout << "Y  Y_test\n";
    for (int i = 0; i < Y_predict.size(); i++) {
      std::cout << Y_predict[i] << "\t" << Y_test[i] << "\n";
    }

    // calculate accuracy of the model
    int true_predict = 0;
    for (int i = 0; i < Y_predict.size(); i++) {
      if (Y_predict[i] == Y_test[i]) true_predict++;
    }

    std::cout << "accuracy: " << true_predict/(double)Y_predict.size() << "\n";
}
