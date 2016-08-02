#ifndef STORM_SOLVER_LINEAREQUATIONSOLVER_H_
#define STORM_SOLVER_LINEAREQUATIONSOLVER_H_

#include <vector>
#include <memory>

#include "src/solver/AbstractEquationSolver.h"

#include "src/storage/SparseMatrix.h"

namespace storm {
    namespace solver {
        
        enum class LinearEquationSolverOperation {
            SolveEquations, MultiplyRepeatedly
        };
        
        /*!
         * An interface that represents an abstract linear equation solver. In addition to solving a system of linear
         * equations, the functionality to repeatedly multiply a matrix with a given vector is provided.
         */
        template<class ValueType>
        class LinearEquationSolver : public AbstractEquationSolver<ValueType> {
        public:
            LinearEquationSolver();
            
            virtual ~LinearEquationSolver() {
                // Intentionally left empty.
            }
            
            virtual void setMatrix(storm::storage::SparseMatrix<ValueType> const& A) = 0;
            virtual void setMatrix(storm::storage::SparseMatrix<ValueType>&& A) = 0;
            
            /*!
             * Solves the equation system A*x = b. The matrix A is required to be square and have a unique solution.
             * The solution of the set of linear equations will be written to the vector x. Note that the matrix A has
             * to be given upon construction time of the solver object.
             *
             * @param x The solution vector that has to be computed. Its length must be equal to the number of rows of A.
             * @param b The right-hand side of the equation system. Its length must be equal to the number of rows of A.
             */
            virtual void solveEquations(std::vector<ValueType>& x, std::vector<ValueType> const& b) const = 0;
            
            /*!
             * Performs on matrix-vector multiplication x' = A*x + b.
             *
             * @param x The input vector with which to multiply the matrix. Its length must be equal
             * to the number of columns of A.
             * @param b If non-null, this vector is added after the multiplication. If given, its length must be equal
             * to the number of rows of A.
             * @param result The target vector into which to write the multiplication result. Its length must be equal
             * to the number of rows of A.
             */
            virtual void multiply(std::vector<ValueType>& x, std::vector<ValueType> const* b, std::vector<ValueType>& result) const = 0;
            
            /*!
             * Performs repeated matrix-vector multiplication, using x[0] = x and x[i + 1] = A*x[i] + b. After
             * performing the necessary multiplications, the result is written to the input vector x. Note that the
             * matrix A has to be given upon construction time of the solver object.
             *
             * @param x The initial vector with which to perform matrix-vector multiplication. Its length must be equal
             * to the number of columns of A.
             * @param b If non-null, this vector is added after each multiplication. If given, its length must be equal
             * to the number of rows of A.
             * @param n The number of times to perform the multiplication.
             */
            void repeatedMultiply(std::vector<ValueType>& x, std::vector<ValueType> const* b, uint_fast64_t n) const;
            
            // Methods related to allocating/freeing auxiliary storage.
            
            /*!
             * Allocates auxiliary memory that can be used to perform the provided operation. Repeated calls to the
             * corresponding function can then be run without allocating/deallocating this memory repeatedly.
             * Note: Since the allocated memory is fit to the currently selected options of the solver, they must not
             * be changed any more after allocating the auxiliary memory until it is deallocated again.
             *
             * @return True iff auxiliary memory was allocated.
             */
            virtual bool allocateAuxMemory(LinearEquationSolverOperation operation) const;
            
            /*!
             * Destroys previously allocated auxiliary memory for the provided operation.
             *
             * @return True iff auxiliary memory was deallocated.
             */
            virtual bool deallocateAuxMemory(LinearEquationSolverOperation operation) const;
            
            /*!
             * If the matrix dimensions changed and auxiliary memory was allocated, this function needs to be called to
             * update the auxiliary memory.
             *
             * @return True iff the auxiliary memory was reallocated.
             */
            virtual bool reallocateAuxMemory(LinearEquationSolverOperation operation) const;
            
            /*!
             * Checks whether the solver has allocated auxiliary memory for the provided operation.
             *
             * @return True iff auxiliary memory was previously allocated (and not yet deallocated).
             */
            virtual bool hasAuxMemory(LinearEquationSolverOperation operation) const;
            
        private:
            /*!
             * Retrieves the row count of the matrix associated with this solver.
             */
            virtual uint64_t getMatrixRowCount() const = 0;
            
            /*!
             * Retrieves the column count of the matrix associated with this solver.
             */
            virtual uint64_t getMatrixColumnCount() const = 0;
            
            // Auxiliary memory for repeated matrix-vector multiplication.
            mutable std::unique_ptr<std::vector<ValueType>> auxiliaryRepeatedMultiplyMemory;
        };
        
        template<typename ValueType>
        class LinearEquationSolverFactory {
        public:
            /*!
             * Creates a new linear equation solver instance with the given matrix.
             *
             * @param matrix The matrix that defines the equation system.
             * @return A pointer to the newly created solver.
             */
            virtual std::unique_ptr<LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const = 0;
            
            /*!
             * Creates a new linear equation solver instance with the given matrix. The caller gives up posession of the
             * matrix by calling this function.
             *
             * @param matrix The matrix that defines the equation system.
             * @return A pointer to the newly created solver.
             */
            virtual std::unique_ptr<LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType>&& matrix) const;
            
            /*!
             * Creates a copy of this factory.
             */
            virtual std::unique_ptr<LinearEquationSolverFactory<ValueType>> clone() const = 0;
        };
        
        template<typename ValueType>
        class GeneralLinearEquationSolverFactory : public LinearEquationSolverFactory<ValueType> {
        public:
            virtual std::unique_ptr<LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const override;
            virtual std::unique_ptr<LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType>&& matrix) const override;
            
            virtual std::unique_ptr<LinearEquationSolverFactory<ValueType>> clone() const override;
            
        private:
            template<typename MatrixType>
            std::unique_ptr<LinearEquationSolver<ValueType>> selectSolver(MatrixType&& matrix) const;
        };
        
#ifdef STORM_HAVE_CARL
        template<>
        class GeneralLinearEquationSolverFactory<storm::RationalNumber> : public LinearEquationSolverFactory<storm::RationalNumber> {
        public:
            virtual std::unique_ptr<LinearEquationSolver<storm::RationalNumber>> create(storm::storage::SparseMatrix<storm::RationalNumber> const& matrix) const override;
            virtual std::unique_ptr<LinearEquationSolver<storm::RationalNumber>> create(storm::storage::SparseMatrix<storm::RationalNumber>&& matrix) const override;
            
            virtual std::unique_ptr<LinearEquationSolverFactory<storm::RationalNumber>> clone() const override;

        private:
            template<typename MatrixType>
            std::unique_ptr<LinearEquationSolver<storm::RationalNumber>> selectSolver(MatrixType&& matrix) const;
        };
        
        template<>
        class GeneralLinearEquationSolverFactory<storm::RationalFunction> : public LinearEquationSolverFactory<storm::RationalFunction> {
        public:
            virtual std::unique_ptr<LinearEquationSolver<storm::RationalFunction>> create(storm::storage::SparseMatrix<storm::RationalFunction> const& matrix) const override;
            virtual std::unique_ptr<LinearEquationSolver<storm::RationalFunction>> create(storm::storage::SparseMatrix<storm::RationalFunction>&& matrix) const override;
            
            virtual std::unique_ptr<LinearEquationSolverFactory<storm::RationalFunction>> clone() const override;

        private:
            template<typename MatrixType>
            std::unique_ptr<LinearEquationSolver<storm::RationalFunction>> selectSolver(MatrixType&& matrix) const;
        };
#endif
    } // namespace solver
} // namespace storm

#endif /* STORM_SOLVER_LINEAREQUATIONSOLVER_H_ */
