#include"Dynamic_tensor.h"
namespace AI {
	Dynamic_tensor::operator parameter_t() const noexcept {

		return _value;
	}
	parameter_t& Dynamic_tensor::operator=(const parameter_t& value) noexcept {
		_value = value;

		return _value;
	}
	void Dynamic_tensor::operator =(Dynamic_tensor&& other) noexcept {
		_value = std::move(other._value);
		_matrices = std::move(other._matrices);
	}
	void Dynamic_tensor::resize(const int& size) noexcept {
		_matrices.resize(size);
	}
	Dynamic_tensor& Dynamic_tensor::operator[](const int& index) noexcept {

		return _matrices[index];
	}
	const Dynamic_tensor& Dynamic_tensor::operator[](const int& index) const noexcept {

		return _matrices[index];
	}
	std::ostream& operator<<(std::ostream& os, const Dynamic_tensor& matrix) {
		os << matrix._value << " " << matrix._matrices.size() << " ";
		for (const Dynamic_tensor& i : matrix._matrices) {
			os << i;
		}

		return os;
	}
	std::istream& operator>>(std::istream& is, Dynamic_tensor& matrix) {
		int size;
		is >> matrix._value >> size;
		matrix._matrices.clear();
		matrix._matrices.resize(size);
		for (int i = 0;i < size;i++) {
			is >> matrix._matrices[i];
		}

		return is;
	}
}