#ifndef STORM_SOLVER_MINMAXLINEAREQUATIONSOLVER_H_
#define STORM_SOLVER_MINMAXLINEAREQUATIONSOLVER_H_

#include <vector>
#include <cstdint>
#include <memory>

#include <boost/optional.hpp>

#include "src/solver/AbstractEquationSolver.h"
#include "src/solver/SolverSelectionOptions.h"
#include "src/storage/sparse/StateType.h"
#include "src/storage/TotalScheduler.h"
#include "src/solver/OptimizationDirection.h"

#include "src/exceptions/InvalidSettingsException.h"

namespace storm {
    namespace storage {
        template<typename T> class SparseMatrix;
    }
    
    namespace solver {
        
        /*!
         * A class representing the interface that all min-max linear equation solvers shall implement.
         */
        template<class ValueType>
        class MinMaxLinearEquationSolver : public AbstractEquationSolver<ValueType> {
        protected:
            MinMaxLinearEquationSolver(OptimizationDirectionSetting direction = OptimizationDirectionSetting::Unset);
        
        public:
            virtual ~MinMaxLinearEquationSolver();

            /*!
             * Solves the equation system x = min/max(A*x + b) given by the parameters. Note that the matrix A has
             * to be given upon construction time of the solver object.
             *
             * @param d For minimum, all the value of a group of rows is the taken as the minimum over all rows and as
             * the maximum otherwise.
             * @param x The solution vector x. The initial values of x represent a guess of the real values to the
             * solver, but may be ignored.
             * @param b The vector to add after matrix-vector multiplication.
             */
            virtual bool solveEquations(OptimizationDirection d, std::vector<ValueType>& x, std::vector<ValueType> const& b) const = 0;
            
            /*!
             * Behaves the same as the other variant of <code>solveEquations</code>, with the distinction that
             * instead of providing the optimization direction as an argument, the internally set optimization direction
             * is used. Note: this method can only be called after setting the optimization direction.
             */
            void solveEquations(std::vector<ValueType>& x, std::vector<ValueType> const& b) const;
            
            /*!
             * Performs (repeated) matrix-vector multiplication with the given parameters, i.e. computes
             * x[i+1] = min/max(A*x[i] + b) until x[n], where x[0] = x. After each multiplication and addition, the
             * minimal/maximal value out of each row group is selected to reduce the resulting vector to obtain the
             * vector for the next iteration. Note that the matrix A has to be given upon construction time of the
             * solver object.
             *
             * @param d For minimum, all the value of a group of rows is the taken as the minimum over all rows and as
             * the maximum otherwise.
             * @param x The initial vector that is to be multiplied with the matrix. This is also the output parameter,
             * i.e. after the method returns, this vector will contain the computed values.
             * @param b If not null, this vector is added after each multiplication.
             * @param n Specifies the number of iterations the matrix-vector multiplication is performed.
             * @param multiplyResult If non-null, this memory is used as a scratch memory. If given, the length of this
             * vector must be equal to the number of rows of A.
             * @return The result of the repeated matrix-vector multiplication as the content of the vector x.
             */
            virtual void repeatedMultiply(OptimizationDirection d, std::vector<ValueType>& x, std::vector<ValueType>* b, uint_fast64_t n = 1) const = 0;
            
            /*!
             * Behaves the same as the other variant of <code>multiply</code>, with the
             * distinction that instead of providing the optimization direction as an argument, the internally set
             * optimization direction is used. Note: this method can only be called after setting the optimization direction.
             */
            virtual void repeatedMultiply(std::vector<ValueType>& x, std::vector<ValueType>* b , uint_fast64_t n) const;

            /*!
             * Sets an optimization direction to use for calls to methods that do not explicitly provide one.
             */
            void setOptimizationDirection(OptimizationDirection direction);

            /*!
             * Unsets the optimization direction to use for calls to methods that do not explicitly provide one.
             */
            void unsetOptimizationDirection();

            /*!
             * Sets whether schedulers are generated when solving equation systems. If the argument is false, the currently
             * stored scheduler (if any) is deleted.
             */
            void setTrackScheduler(bool trackScheduler = true);

            /*!
             * Retrieves whether this solver is set to generate schedulers.
             */
            bool isTrackSchedulerSet() const;

            /*!
             * Retrieves whether the solver generated a scheduler.
             */
            bool hasScheduler() const;

            /*!
             * Retrieves the generated scheduler. Note: it is only legal to call this function if a scheduler was generated.
             */
            storm::storage::TotalScheduler const& getScheduler() const;

            /*!
             * Retrieves the generated scheduler and takes ownership of it. Note: it is only legal to call this function
             * if a scheduler was generated and after a call to this method, the solver will not contain the scheduler
             * any more (i.e. it is illegal to call this method again until a new scheduler has been generated).
             */
            std::unique_ptr<storm::storage::TotalScheduler> getScheduler();

            /**
             * Gets the precision after which the solver takes two numbers as equal.
             *
             * @see getRelative()
             */
            virtual ValueType getPrecision() const = 0;

            /**
             *  Gets whether the precision is taken to be absolute or relative
             */
            virtual bool getRelative() const = 0;

            /*
             * Resets the auxiliary data that has been stored during previous calls of this solver
             */
            virtual void resetAuxiliaryData() const;


        protected:
            /// The optimization direction to use for calls to functions that do not provide it explicitly. Can also be unset.
            OptimizationDirectionSetting direction;

            /// Whether we generate a scheduler during solving.
            bool trackScheduler;

            /// The scheduler (if it could be successfully generated).
            mutable boost::optional<std::unique_ptr<storm::storage::TotalScheduler>> scheduler;
        };

        template<typename ValueType>
        class MinMaxLinearEquationSolverFactory {
        public:
            MinMaxLinearEquationSolverFactory(bool trackScheduler = false);

            virtual std::unique_ptr<MinMaxLinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const = 0;
            virtual std::unique_ptr<MinMaxLinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType>&& matrix) const;

            void setTrackScheduler(bool value);
            bool isTrackSchedulerSet() const;

        private:
            bool trackScheduler;
        };

        template<typename ValueType>
        class GeneralMinMaxLinearEquationSolverFactory : public MinMaxLinearEquationSolverFactory<ValueType> {
        public:
            GeneralMinMaxLinearEquationSolverFactory(bool trackScheduler = false);

            virtual std::unique_ptr<MinMaxLinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const override;
            virtual std::unique_ptr<MinMaxLinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType>&& matrix) const override;

        private:
            template<typename MatrixType>
            std::unique_ptr<MinMaxLinearEquationSolver<ValueType>> selectSolver(MatrixType&& matrix) const;
        };
        
    } // namespace solver
} // namespace storm

#endif /* STORM_SOLVER_MINMAXLINEAREQUATIONSOLVER_H_ */
