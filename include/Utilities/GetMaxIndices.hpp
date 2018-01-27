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
  	else if (std::abs(v[i] - maxValue) < 1e-99){
  		maxIndices.push_back(i);
  	}
  }
  return maxIndices;
}