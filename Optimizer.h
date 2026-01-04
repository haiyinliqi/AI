#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_
#include<vector>
#include<cmath>
#include"ai_type.h"
#include"Optimization_strategy.h"
namespace AI {
	class Optimizer {//优化器类
	protected:
		struct _One_ai_group {//单个神经网络
			struct _One_parameter_matrix {//单个参数及导数矩阵
				parameter_t** _parameters, ** _def;//参数矩阵和导数矩阵
				int _width, _height;//矩阵宽度、高度
				_One_parameter_matrix(parameter_t** parameters, parameter_t** def, int height, int width) noexcept;
			};
			//释放内存
			virtual void del() noexcept;
		};
		std::vector<Optimization_strategy*> _optimizationStrategies;//优化策略集合
		int _max_id = -1;//管理的最大神经网络组id
		//使用策略
		void use_strategy(parameter_t parameter, parameter_t& def) noexcept;
		//初始化
		virtual int init() noexcept = 0;
		//注册参数
		virtual void file_parameters(int id, parameter_t** parameters, parameter_t** def, int height, int width) noexcept = 0;
		//调参
		virtual void register_parameters(int id, parameter_t lr) noexcept = 0;
		//移除神经网络
		virtual void del(int id) noexcept = 0;
	public:
		//添加优化策略并自动排序
		void add_strategy(Optimization_strategy* Optimization_strategy) noexcept;
		virtual ~Optimizer() noexcept = default;
		friend class Layer;
		friend class Ai;
	};
	class SGD :public Optimizer {//随机梯度下降优化器类
	protected:
		struct _SGD_one_ai_group :public _One_ai_group {
			struct _SGD_one_parameter_matrix :public _One_parameter_matrix {
				_SGD_one_parameter_matrix(parameter_t** parameter, parameter_t** def, int height, int width) noexcept;
			};
			std::vector<_SGD_one_parameter_matrix> _parameter_matrices;//所有参数组
		};
		std::vector< _SGD_one_ai_group> _all_ai;//所有神经网络组
		int init() noexcept override;
		void file_parameters(int id, parameter_t** parameters, parameter_t** def, int height, int width) noexcept override;
		void register_parameters(int id, parameter_t lr) noexcept override;
		void del(int id) noexcept override;
	};
	class Adam :public Optimizer {//自适应矩估计优化器类
	protected:
		struct _Adam_one_ai_group :public _One_ai_group {
			struct _Adam_one_parameter_matrix :public _One_parameter_matrix {
				parameter_t** _m, ** _v;//参数一阶矩、二阶矩
				_Adam_one_parameter_matrix(parameter_t** parameter, parameter_t** def, int height, int width) noexcept;
			};
			std::vector<_Adam_one_parameter_matrix> _parameter_matrices;//所有参数组
			int _t = 0;//调参次数
			bool is_del = false;//是否已释放内存
			void del() noexcept override;
		};
		std::vector< _Adam_one_ai_group> _all_ai;//所有神经网络组
		parameter_t _beta1, _beta2, _epsilon;//一阶矩衰减率、二阶矩衰减率、数值稳定项
		int init() noexcept override;
		void file_parameters(int id, parameter_t** parameters, parameter_t** def, int height, int width) noexcept override;
		void register_parameters(int id, parameter_t lr) noexcept override;
		void del(int id) noexcept override;
	public:
		Adam(parameter_t beta1, parameter_t beta2, parameter_t epsilon = 1e-18) noexcept;
		~Adam() noexcept;
	};
}
#endif