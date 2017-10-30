template <class T>
std::vector<T> getMaxIndices ( std::vector<T> v) {
  T maxValue = v[0];
  std::vector<T> maxIndices = {0};
  for (size_t i = 1; i < v.size(); ++i){
  	if (v[i] > maxValue){
  		maxIndices.clear();
  		maxValue = v[i];
  		maxIndices.push_back(i);
  	}
  	else if (v[i] == maxValue){
  		maxIndices.push_back(i);
  	}
  }
  return maxIndices;
}