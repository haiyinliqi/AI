#ifndef _LAYER_H_
#define _LAYER_H_ true
#include<random>
#include"ai_type.h"
#include"Dynamic_tensor.h"
#include"ActivationFunction.h"
#include"Optimizer.h"
namespace AI {
	extern std::mt19937 gen;//随机数生成器
	extern std::uniform_real_distribution<parameter_t> dis;//-1到1的映射器
	class Layer {//单层神经网络类
	protected:
		parameter_t* _input, * _input_def, * _output, * _output_def;//输入、输入导数、输出、输出导数
		int _input_num, _output_num;//上一层神经元数量、本层神经元数量
		bool _only_compute;//只计算标识符
		Layer(int input_num, int output_num, parameter_t* input, parameter_t*& output, parameter_t* input_def, parameter_t*& output_def, bool only_compute) noexcept;
		//将参数注册进优化器（给派生类提供的接口）
		void file_parameters(parameter_t** parameters, parameter_t** def, int height, int width, Optimizer* optimizer, int optimizer_id) noexcept;
		//将参数注册进优化器
		virtual void file_parameters(Optimizer* optimizer, int optimizer_id) noexcept = 0;
		//设置参数
		virtual void set_parameters(const Dynamic_tensor& parameters) noexcept = 0;
		//计算输出
		virtual void calculate_output() noexcept = 0;
		//反向传播
		virtual void register_parameters() noexcept = 0;
		//获取参数
		virtual Dynamic_tensor get_parameters() const noexcept = 0;
		virtual ~Layer() noexcept;
		friend class Ai;
		friend class Compute_ai;
		friend class Ai_builder;
		friend class Compute_ai_builder;
	};
	enum class Layer_kinds {
		Fully_connected_layer, Convolutional_layer
	};
	template<typename T>
	class Fully_connected_layer :public Layer {//全连接层类
	protected:
		T* _activationFunction;//激活函数
		parameter_t** _weights, ** _weights_def, * _biases, * _biases_def;//权重、权重导数、偏置项、偏置项导数
		Fully_connected_layer(int input_num, int neuron_num, parameter_t* input, parameter_t*& output, parameter_t* input_def, parameter_t*& output_def, bool only_compute) noexcept :
			Layer(input_num, neuron_num, input, output, input_def, output_def, only_compute) {
			_weights = new parameter_t * [_output_num];
			_biases = new parameter_t[_output_num];
			if (!_only_compute) {
				_weights_def = new parameter_t * [_output_num];
				_biases_def = new parameter_t[_output_num];
			}
			for (int i = 0;i < _output_num;i++) {
				_weights[i] = new parameter_t[_input_num];
				if (!_only_compute) {
					_weights_def[i] = new parameter_t[_input_num];
				}
				for (int j = 0;j < _input_num;j++) {
					_weights[i][j] = dis(gen);
					if (!_only_compute) {
						_weights_def[i][j] = 0;
					}
				}
				_biases[i] = dis(gen);
				if (!_only_compute) {
					_biases_def[i] = 0;
				}
			}
			_activationFunction = new T(_output_num, _output, output_def);
		}
		void file_parameters(Optimizer* optimizer, int optimizer_id) noexcept {
			Layer::file_parameters(&_biases, &_biases_def, 1, _output_num, optimizer, optimizer_id);
			Layer::file_parameters(_weights, _weights_def, _output_num, _input_num, optimizer, optimizer_id);
		}
		void set_parameters(const Dynamic_tensor& parameters) noexcept {
			for (int i = 0;i < _output_num;i++) {
				for (int j = 0;j < _input_num;j++) {
					_weights[i][j] = parameters[i][j];
				}
				_biases[i] = parameters[i];
			}
		}
		void calculate_output() noexcept {
			if (_output_num > 5000 && _output_num * _input_num > 10000000) {
#pragma omp parallel for schedule(static)
				for (int i = 0;i < _output_num;i++) {
					_output[i] = 0;
					for (int j = 0;j < _input_num;j++) {
						_output[i] += _input[j] * _weights[i][j];
					}
					_output[i] += _biases[i];
				}
			}
			else {
				for (int i = 0;i < _output_num;i++) {
					_output[i] = 0;
					for (int j = 0;j < _input_num;j++) {
						_output[i] += _input[j] * _weights[i][j];
					}
					_output[i] += _biases[i];
				}
			}
			_activationFunction->compute_output();
		}
		void register_parameters() noexcept {
			_activationFunction->compute_def();
			if (_output_num > 5000 && _output_num * _input_num > 10000000) {
#pragma omp parallel for schedule(static)
				for (int i = 0;i < _output_num;i++) {
					for (int j = 0;j < _input_num;j++) {
						_weights_def[i][j] += _output_def[i] * _input[j];
					}
					_biases_def[i] += _output_def[i];
				}
			}
			else {
				for (int i = 0;i < _output_num;i++) {
					for (int j = 0;j < _input_num;j++) {
						_weights_def[i][j] += _output_def[i] * _input[j];
					}
					_biases_def[i] += _output_def[i];
				}
			}
			if (_input_num > 5000 && _input_num * _output_num > 10000000) {
#pragma omp parallel for schedule(static)
				for (int i = 0;i < _input_num;i++) {
					_input_def[i] = 0;
					for (int j = 0;j < _output_num;j++) {
						_input_def[i] += _output_def[j] * _weights[j][i];
					}
				}
			}
			else {
				for (int i = 0;i < _input_num;i++) {
					_input_def[i] = 0;
					for (int j = 0;j < _output_num;j++) {
						_input_def[i] += _output_def[j] * _weights[j][i];
					}
				}
			}
		}
		Dynamic_tensor get_parameters() const noexcept {
			Dynamic_tensor ans;
			ans.resize(_output_num);
			for (int i = 0;i < _output_num;i++) {
				ans[i].resize(_input_num);
				for (int j = 0;j < _input_num;j++) {
					ans[i][j] = _weights[i][j];
				}
				ans[i] = _biases[i];
			}

			return ans;
		}
		~Fully_connected_layer() noexcept {
			for (int i = 0;i < _output_num;i++) {
				delete[] _weights[i];
				if (!_only_compute) {
					delete[] _weights_def[i];
				}
			}
			delete[] _weights;
			delete[] _biases;
			if (!_only_compute) {
				delete[] _weights_def;
				delete[] _biases_def;
			}
			delete _activationFunction;
		}
		friend class Ai_builder;
		friend class Compute_ai_builder;
	};
	template<typename T>
	class Convolutional_layer :public Layer {//卷积层类
	protected:
		T* _activationFunction;//激活函数
		parameter_t**** _weights, **** _weights_def, * _biases, * _biases_def;//权重、权重导数、偏置项、偏置项导数
		//  步长     输入高度       输入宽度      输入通道数          卷积核数     卷积核大小    输出高度        输出宽度
		int _stride, _input_height, _input_width, _input_channelsint, _kernel_num, _kernel_size, _output_height, _output_width;
		//将三维数据映射为数组下标
#define to_index(i,j,l,height,width) (i)*(height)*(width)+(j)*(width)+(l)
		//使用三维索引访问输入
#define to_input(i,j,l) _input[to_index(i,j,l,_input_height,_input_width)]
	//使用三维索引访问输出
#define to_output(i,j,l) _output[to_index(i,j,l,_output_height,_output_width)]
	//使用三维索引访问输入导数
#define to_input_def(i,j,l) _input_def[to_index(i,j,l,_input_height,_input_width)]
	//使用三维索引访问输出导数
#define to_output_def(i,j,l)_output_def[to_index(i,j,l,_output_height,_output_width,_kernel_num)]
		Convolutional_layer(int input_height, int input_width, int input_channelsint, int kernel_num, int kernel_size, int stride, parameter_t* input, parameter_t*& output, parameter_t* input_def, parameter_t*& output_def, bool only_compute) noexcept :
			_input_height(input_height), _input_width(input_width), _input_channelsint(input_channelsint), _kernel_num(kernel_num), _kernel_size(kernel_size), _stride(stride), _output_height((input_height - kernel_size) / stride + 1), _output_width((input_width - kernel_size) / stride + 1),
			Layer(input_height* input_width* input_channelsint, ((input_height - kernel_size) / stride + 1)* ((input_width - kernel_size) / stride + 1)* kernel_num, input, output, input_def, output_def, only_compute) {
			_weights = new parameter_t * **[_kernel_num];
			_biases = new parameter_t[_kernel_num];
			if (!_only_compute) {
				_weights_def = new parameter_t * **[_kernel_num];
				_biases_def = new parameter_t[_kernel_num];
			}
			for (int i = 0;i < _kernel_num;i++) {
				_weights[i] = new parameter_t * *[_input_channelsint];
				if (!_only_compute) {
					_weights_def[i] = new parameter_t * *[_input_channelsint];
				}
				for (int j = 0;j < _input_channelsint;j++) {
					_weights[i][j] = new parameter_t * [_kernel_size];
					if (!_only_compute) {
						_weights_def[i][j] = new parameter_t * [_kernel_size];
					}
					for (int l = 0;l < _kernel_size;l++) {
						_weights[i][j][l] = new parameter_t[_kernel_size];
						if (!_only_compute) {
							_weights_def[i][j][l] = new parameter_t[_kernel_size];
						}
						for (int k = 0;k < _kernel_size;k++) {
							if (!_only_compute) {
								_weights[i][j][l][k] = dis(gen);
							}
							if (!_only_compute) {
								_weights_def[i][j][l][k] = 0;
							}
						}
					}
				}
				_biases[i] = dis(gen);
				if (!_only_compute) {
					_biases_def[i] = 0;
				}
			}
			_activationFunction = new T(_output_num, _output, output_def);
		}
		void file_parameters(Optimizer* optimizer, int optimizer_id) noexcept {
			Layer::file_parameters(&_biases, &_biases_def, 1, _kernel_num, optimizer, optimizer_id);
			for (int i = 0;i < _kernel_num;i++) {
				for (int j = 0;j < _input_channelsint;j++) {
					Layer::file_parameters(_weights[i][j], _weights_def[i][j], _kernel_size, _kernel_size, optimizer, optimizer_id);
				}
			}
		}
		void set_parameters(const Dynamic_tensor& parameters) noexcept {
			for (int i = 0;i < _kernel_num;i++) {
				for (int j = 0;j < _input_channelsint;j++) {
					for (int l = 0;l < _kernel_size;l++) {
						for (int k = 0;k < _kernel_size;k++) {
							_weights[i][j][l][k] = parameters[i][j][l][k];
						}
					}
				}
				_biases[i] = parameters[i];
			}
		}
		void calculate_output() noexcept {
			for (int i = 0;i < _kernel_num;i++) {
				for (int j = 0;j + _kernel_size - 1 < _input_height;j += _stride) {
					for (int l = 0;l + _kernel_size - 1 < _input_width;l += _stride) {
						to_output(i, j / _stride, l / _stride) = _biases[i];
						for (int k = 0;k < _input_channelsint;k++) {
							for (int s = 0;s < _kernel_size;s++) {
								for (int r = 0;r < _kernel_size;r++) {
									to_output(i, j / _stride, l / _stride) += to_input(k, j + s, l + r) * _weights[i][k][s][r];
								}
							}
						}
					}
				}
			}
			_activationFunction->compute_output();
		}
		void register_parameters() noexcept {
			_activationFunction->compute_def();
			for (int i = 0;i < _input_channelsint;i++) {
				for (int j = 0;j < _input_height;j++) {
					for (int l = 0;l < _input_width;l++) {
						to_input_def(i, j, l) = 0;
					}
				}
			}
			for (int i = 0;i < _kernel_num;i++) {
				for (int j = 0;j + _kernel_size - 1 < _input_height;j += _stride) {
					for (int l = 0;l + _kernel_size - 1 < _input_width;l += _stride) {
						to_output(i, j / _stride, l / _stride) = _biases[i];
						for (int k = 0;k < _input_channelsint;k++) {
							for (int s = 0;s < _kernel_size;s++) {
								for (int r = 0;r < _kernel_size;r++) {
									to_input_def(k, j + s, l + r) += to_output_def(i, j / _stride, l / _stride) * _weights[i][k][s][r];
									_weights_def[i][k][s][r] += to_output_def(i, j / _stride, l / _stride) * to_input(k, j + s, l + r);
								}
							}
						}
					}
				}
			}
		}
		Dynamic_tensor get_parameters() const noexcept {
			Dynamic_tensor ans;
			ans.resize(_kernel_num);
			for (int i = 0;i < _kernel_num;i++) {
				ans[i].resize(_input_channelsint);
				for (int j = 0;j < _input_channelsint;j++) {
					ans[i][j].resize(_kernel_size);
					for (int l = 0;l < _kernel_size;l++) {
						ans[i][j][l].resize(_kernel_size);
						for (int k = 0;k < _kernel_size;k++) {
							ans[i][j][l][k] = _weights[i][j][l][k];
						}
					}
				}
				ans[i] = _biases[i];
			}

			return ans;
		}
		~Convolutional_layer() noexcept {
			for (int i = 0;i < _kernel_num;i++) {
				for (int j = 0;j < _input_channelsint;j++) {
					for (int l = 0;l < _kernel_size;l++) {
						delete[] _weights[i][j][l];
						if (!_only_compute) {
							delete[] _weights_def[i][j][l];
						}
					}
					delete[] _weights[i][j];
					if (!_only_compute) {
						delete[] _weights_def[i][j];
					}
				}
				delete[] _weights[i];
				if (!_only_compute) {
					delete[] _weights_def[i];
				}
			}
			delete[] _weights;
			delete[] _biases;
			if (!_only_compute) {
				delete[] _weights_def;
				delete[] _biases_def;
			}
			delete _activationFunction;
		}
		friend class Ai_builder;
		friend class Compute_ai_builder;
#undef to_index
#undef to_input
#undef to_output
#undef to_input_def
#undef to_output_def
	};
}
#endif