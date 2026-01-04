/*
待实现功能：
根据Dynamic_tensor构造神经网络
BatchNormLayer
卷积层
反卷积层
神经网络连接器
使用NPU或GPU加速运算
ONNX模型序列化功能
*/
/*
过程中可选择性实现：
debug宏
*/
#ifndef _AI_H_
#define _AI_H_ true
#include<algorithm>
#include<fstream>
#include<windows.h>
#include"ai_type.h"
#include"Layer.h"
#include"LossFunction.h"
namespace AI {
	class Ai {//人工神经网络类
	protected:
		struct _Sample {//训练样本
			parameter_t* _input, * _expectation;//输入、输出
		};
		LossFunction* _lossFunction;//损失函数
		Optimizer* _optimizer;//优化器
		parameter_t** _output, ** _def;//输出、神经元整体导数
		int _input_num, _output_num, _size, _cnt;//输入个数、输出个数、神经网络层数、样本数量
		Layer** _layers;//每层神经网络指针
		_Sample* _samples;//训练样本
		int _optimizer_id;//优化器的id
		Ai(int size, Layer** layers, LossFunction_kinds lossfuncion_type, Optimizer* optimizer, parameter_t** output, parameter_t** def, int input_num, int output_num) noexcept;
		//将参数注册进优化器
		void file_parameters() noexcept;
	public:
		//设置样本
		void set_samples(std::string file, int cnt, bool have_expectation = true) noexcept;
		//设置参数
		void set_parameters(const Dynamic_tensor& parameters) noexcept;
		//设置优化器
		void set_optimizer(Optimizer* optimizer) noexcept;
		//进行单次计算
		parameter_t* compute(parameter_t* input) const noexcept;
		//单次调参
		parameter_t register_parameters(parameter_t* expectation, parameter_t lr) noexcept;
		//按次数训练
		void train(int n, parameter_t lr) noexcept;
		//按损失训练（附加最大训练次数）
		void train(parameter_t expect, int maxn, parameter_t lr) noexcept;
		//获取参数
		Dynamic_tensor get_parameters() const noexcept;
		~Ai() noexcept;
		friend class Ai_builder;
	};
	class Compute_ai {//只计算人工神经网络类
	protected:
		parameter_t** _output;//输出
		int _input_num, _output_num, _size;//输入个数、输出个数、神经网络层数
		Layer** _layers;//每层神经网络指针
		Compute_ai(int size, Layer** layers, parameter_t** output, int input_num, int output_num) noexcept;
	public:
		//设置参数
		void set_parameters(const Dynamic_tensor& parameters) noexcept;
		//进行单次计算
		parameter_t* compute(parameter_t* input) const noexcept;
		//获取参数
		const Dynamic_tensor& get_parameters() const noexcept;
		~Compute_ai() noexcept;
		friend class Compute_ai_builder;
	};
	class Ai_builder {//人工神经网络建造者
	protected:
		Ai_builder() noexcept = delete;
		inline static std::vector<Layer*> _layers;//所有层
		inline static std::vector<parameter_t*> _output, _def;//输出、导数
		inline static Optimizer* _optimizer = nullptr;//优化器
		inline static int _ai_input_num = 0, _input_num = 0;//神经网络输入数量、当前层输入数量
		class _Set_lossFunction {//设置损失函数类
		public:
			//设置损失函数并构造神经网络
			static Ai* set_lossFunction(LossFunction_kinds lossFunction_type) noexcept;
		};
		class _Add_layer;
		class _Add_convolutional_layer;
		class _Add_layer {//添加层、设置优化器类
		public:
			//添加全连接层
			template<typename T>
			static _Add_layer* add_fully_connected_layer(int neuron_num) noexcept {
				Ai_builder::_output.push_back(nullptr);
				Ai_builder::_def.push_back(nullptr);
				Ai_builder::_layers.push_back(new Fully_connected_layer<T>(Ai_builder::_input_num, neuron_num, Ai_builder::_output[Ai_builder::_layers.size()], Ai_builder::_output[Ai_builder::_layers.size() + 1], Ai_builder::_def[Ai_builder::_layers.size()], Ai_builder::_def[Ai_builder::_layers.size() + 1], false));
				Ai_builder::_input_num = neuron_num;

				return nullptr;
			}
			//添加卷积层
			template<typename T>
			static _Add_convolutional_layer* add_convolutional_layer(int kernel_num, int kernel_size, int stride, int input_height, int input_width) noexcept {
				if (Ai_builder::_input_num % (input_height * input_width) != 0) {
					Ai_builder::_layers.clear();
					Ai_builder::_output.clear();
					Ai_builder::_def.clear();

					throw 0;
				}
				Ai_builder::_output.push_back(nullptr);
				Ai_builder::_def.push_back(nullptr);
				Ai_builder::_layers.push_back(new Convolutional_layer<T>(input_height, input_width, Ai_builder::_input_num / input_height / input_width, kernel_num, kernel_size, stride, Ai_builder::_output[Ai_builder::_layers.size()], Ai_builder::_output[Ai_builder::_layers.size() + 1], Ai_builder::_def[Ai_builder::_layers.size()], Ai_builder::_def[Ai_builder::_layers.size() + 1], false));
				input_height = ((input_height - kernel_size) / stride + 1);
				input_width = ((input_width - kernel_size) / stride + 1);
				Ai_builder::_Add_convolutional_layer::_input_height = input_height;
				Ai_builder::_Add_convolutional_layer::_input_width = input_width;
				Ai_builder::_input_num = input_height * input_width * kernel_num;

				return nullptr;
			}
			//设置优化器
			static _Set_lossFunction* set_optimizer(Optimizer* optimizer) noexcept;
		};
		class _Add_convolutional_layer :public _Add_layer {//添加卷积层类
		protected:
			inline static int _input_height, _input_width;
		public:
			//添加卷积层
			template<typename T>
			static _Add_convolutional_layer* add_convolutional_layer(int kernel_num, int kernel_size, int stride) noexcept {
				Ai_builder::_output.push_back(nullptr);
				Ai_builder::_def.push_back(nullptr);
				Ai_builder::_layers.push_back(new Convolutional_layer<T>(_input_height, _input_width, Ai_builder::_input_num / _input_height / _input_width, kernel_num, kernel_size, stride, Ai_builder::_output[Ai_builder::_layers.size()], Ai_builder::_output[Ai_builder::_layers.size() + 1], Ai_builder::_def[Ai_builder::_layers.size()], Ai_builder::_def[Ai_builder::_layers.size() + 1], false));
				_input_height = ((_input_height - kernel_size) / stride + 1);
				_input_width = ((_input_width - kernel_size) / stride + 1);
				Ai_builder::_input_num = _input_height * _input_width * kernel_num;

				return nullptr;
			}
			friend class _Add_layer;
		};
		class _Set_input {//设置输入数量类
		public:
			//设置输入数量
			static _Add_layer* set_input_num(int input_num)noexcept;
		};
	public:
		static _Set_input* init() noexcept;
	};
	//环境类型、动作类型
	template<typename T, typename U>
	class Reward_environment {//带奖励的环境
	protected:
		struct _Individual {//单个个体
			Ai* _ai;//神经网络
			parameter_t* (*_change_to_input)(const T&, const U&);//在当前环境下神经网络的输入
			parameter_t (*_perform)(T&, const U&);//在当前环境下执行动作并返回奖励
			std::pair<U*, int>(*_find_performance)(const T&);//在当前环境下所有可执行动作
			parameter_t _lr, _gamma, _explorationRate;//学习率、对未来奖励的重视程度、探索率
			_Individual(Ai* ai, parameter_t* (*change_to_input)(const T&, const U&), parameter_t (*perform)(T&, const U&), std::pair<U*, int>(*find_performance)(const T&), parameter_t lr, parameter_t gamma, parameter_t explorationRate) noexcept :
				_ai(ai), _change_to_input(change_to_input), _perform(perform), _find_performance(find_performance), _lr(lr), _gamma(gamma), _explorationRate(explorationRate) {
			}
		};
		std::vector<_Individual> _all_ai;//全部个体
	public:
		//在环境中加入新个体
		void add_species(Ai* ai, parameter_t* (*change_to_input)(const T&, const U&), parameter_t (*perform)(T&, const U&), std::pair<U*, int>(*find_performance)(const T&), parameter_t lr, parameter_t gamma, parameter_t explorationRate) noexcept {
			_all_ai.emplace_back(ai, change_to_input, perform, find_performance, lr, gamma, explorationRate);
		}
		//设置个体的学习率
		void set_lr(int id, parameter_t lr) noexcept {
			_all_ai[id]._lr = lr;
		}
		//获取个体的学习率
		parameter_t get_lr(int id) const noexcept {

			return _all_ai[id]._lr;
		}
		//设置个体的探索率
		void set_explorationRate(int id, parameter_t explorationRate) noexcept {
			_all_ai[id]._explorationRate = explorationRate;
		}
		//获取个体的探索率
		parameter_t get_explorationRate(int id) const noexcept {

			return _all_ai[id]._explorationRate;
		}
		//让所有体在环境中训练
		void train(T& environment, int num) noexcept {
			struct _Sample {
				T _now_environment;//当前环境
				U _performance;//动作
				parameter_t _now_reward, _future_reward;//当前奖励、未来奖励
				_Sample() {
				}
			};
			_Sample** samples = new _Sample * [num];
			for (int i = 0;i < num;i++) {
				samples[i] = new _Sample[_all_ai.size()];
			}
			for (int i = 0;i <= num;i++) {
				Sleep(200);
				//system("cls");
				for (int j = 0;j < _all_ai.size();j++) {
					std::pair<U*, int> all_choice = _all_ai[j]._find_performance(environment);
					U* performances = all_choice.first;
					int cnt = all_choice.second;
					parameter_t max_reward = std::numeric_limits<parameter_t>::lowest();
					U performance;
					for (int l = 0;l < cnt;l++) {
						parameter_t* input = _all_ai[j]._change_to_input(environment, performances[l]);
						parameter_t* ans = _all_ai[j]._ai->compute(input);
						//std::cout << ans[0] << '\n';
						//Sleep(5000);
						if (ans[0] > max_reward) {
							max_reward = ans[0];
							performance = performances[l];
						}
						delete[] input;
						delete[] ans;
					}
					if (dis(gen) / 2 + 0.5 <= _all_ai[j]._explorationRate) {
						int l = (int)((dis(gen) / 2 + 0.5) * cnt) % cnt;
						performance = performances[l];
					}
					//std::cout << performance << std::endl;
					//Sleep(5000);
					if (i < num) {
						samples[i][j]._now_environment = environment;
						samples[i][j]._performance = performance;
						samples[i][j]._now_reward = _all_ai[j]._perform(environment, performance);
					}
					if (i > 0) {
						samples[i - 1][j]._future_reward = max_reward;
					}
					delete[] performances;
				}
			}
			std::shuffle(samples, samples + num, gen);
			parameter_t loss = 0;
			for (int i = 0;i < num;i++) {
				for (int j = 0;j < _all_ai.size();j++) {
					parameter_t* input = _all_ai[j]._change_to_input(samples[i][j]._now_environment, samples[i][j]._performance);
					delete[] _all_ai[j]._ai->compute(input);
					delete[] input;
					parameter_t expectation[1] = { samples[i][j]._now_reward + _all_ai[j]._gamma * samples[i][j]._future_reward};
					loss += _all_ai[j]._ai->register_parameters(expectation, _all_ai[j]._lr);
				}
			}
			loss /= num;
			std::system("cls");
			std::cout << loss << std::endl;
			for (int i = 0;i < num;i++) {
				delete[] samples[i];
			}
			delete[] samples;
		}
	};
}
#endif
