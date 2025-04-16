#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../Decision_Tree.h"
#include <iostream>
#include <vector>
#include <climits>

TEST_CASE("TEST DECISION TREE FIT()")
{
    std::fstream training_file ("../training.dat", std::ios::in);
    std::vector<std::vector<int>> X;
    std::vector<int> Y;

    // read data from training.dat
    while (!training_file.eof()) {
          std::string line;
          std::getline(training_file, line);
          std::vector<int> row;
          int count = 0;
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

    std::fstream test_file ("../testing.dat", std::ios::in);
    std::vector<std::vector<int>> X_test;
    std::vector<int> Y_test;

    while (!test_file.eof()) {
      std::string line;
      std::getline(test_file, line);
      std::vector<int> row;
      int count = 0;

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

    DecisionTree tree = DecisionTree(INT_MAX, 5, 0.03, 0.1);

    tree.fit(X, Y);

    std::vector<int> Y_predict = tree.predict(X_test);

    std::cout << "Y_predict  Y_test\n";
    for (int i = 0; i < Y_predict.size(); i++) {
      std::cout << Y_predict[i] << "\t" << Y_test[i] << "\n";
    }

    int true_predict = 0;
    for (int i = 0; i < Y_predict.size(); i++) {
      if (Y_predict[i] == Y_test[i]) true_predict++;
    }

    std::cout << "accuracy: " << true_predict/(double)Y_predict.size() << "\n";
}
