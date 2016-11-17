#ifndef STORM_SOLVER_GMMXXLINEAREQUATIONSOLVER_H_
#define STORM_SOLVER_GMMXXLINEAREQUATIONSOLVER_H_

#include <ostream>

#include "src/storm/utility/gmm.h"

#include "LinearEquationSolver.h"

namespace storm {
    namespace solver {

        template<typename ValueType>
        class GmmxxLinearEquationSolverSettings {
        public:
            // An enumeration specifying the available preconditioners.
            enum class Preconditioner {
                Ilu, Diagonal, None
            };

            
            friend std::ostream& operator<<(std::ostream& out, Preconditioner const& preconditioner) {
                switch (preconditioner) {
                    case GmmxxLinearEquationSolverSettings<ValueType>::Preconditioner::Ilu: out << "ilu"; break;
                    case GmmxxLinearEquationSolverSettings<ValueType>::Preconditioner::Diagonal: out << "diagonal"; break;
                    case GmmxxLinearEquationSolverSettings<ValueType>::Preconditioner::None: out << "none"; break;
                }
                return out;
            }

            // An enumeration specifying the available solution methods.
            enum class SolutionMethod {
                Bicgstab, Qmr, Gmres, Jacobi
            };
            
            friend std::ostream& operator<<(std::ostream& out, SolutionMethod const& method) {
                switch (method) {
                    case GmmxxLinearEquationSolverSettings<ValueType>::SolutionMethod::Bicgstab: out << "BiCGSTAB"; break;
                    case GmmxxLinearEquationSolverSettings<ValueType>::SolutionMethod::Qmr: out << "QMR"; break;
                    case GmmxxLinearEquationSolverSettings<ValueType>::SolutionMethod::Gmres: out << "GMRES"; break;
                    case GmmxxLinearEquationSolverSettings<ValueType>::SolutionMethod::Jacobi: out << "Jacobi"; break;
                }
                return out;
            }

            GmmxxLinearEquationSolverSettings();
            
            void setSolutionMethod(SolutionMethod const& method);
            void setPreconditioner(Preconditioner const& preconditioner);
            void setPrecision(ValueType precision);
            void setMaximalNumberOfIterations(uint64_t maximalNumberOfIterations);
            void setRelativeTerminationCriterion(bool value);
            void setNumberOfIterationsUntilRestart(uint64_t restart);
         
            SolutionMethod getSolutionMethod() const;
            Preconditioner getPreconditioner() const;
            ValueType getPrecision() const;
            uint64_t getMaximalNumberOfIterations() const;
            bool getRelativeTerminationCriterion() const;
            uint64_t getNumberOfIterationsUntilRestart() const;
            
        private:
            // The method to use for solving linear equation systems.
            SolutionMethod method;
            
            // The required precision for the iterative methods.
            double precision;
            
            // The maximal number of iterations to do before iteration is aborted.
            uint_fast64_t maximalNumberOfIterations;
            
            // The preconditioner to use when solving the linear equation system.
            Preconditioner preconditioner;
            
            // Sets whether the relative or absolute error is to be considered for convergence detection. Note that this
            // only applies to the Jacobi method for this solver.
            bool relative;
            
            // A restart value that determines when restarted methods shall do so.
            uint_fast64_t restart;
        };
        
        /*!
         * A class that uses the gmm++ library to implement the LinearEquationSolver interface.
         */
        template<typename ValueType>
        class GmmxxLinearEquationSolver : public LinearEquationSolver<ValueType> {
        public:
            GmmxxLinearEquationSolver(storm::storage::SparseMatrix<ValueType> const& A, GmmxxLinearEquationSolverSettings<ValueType> const& settings = GmmxxLinearEquationSolverSettings<ValueType>());
            GmmxxLinearEquationSolver(storm::storage::SparseMatrix<ValueType>&& A, GmmxxLinearEquationSolverSettings<ValueType> const& settings = GmmxxLinearEquationSolverSettings<ValueType>());
            
            virtual void setMatrix(storm::storage::SparseMatrix<ValueType> const& A) override;
            virtual void setMatrix(storm::storage::SparseMatrix<ValueType>&& A) override;
            
            virtual bool solveEquations(std::vector<ValueType>& x, std::vector<ValueType> const& b) const override;
            virtual void multiply(std::vector<ValueType>& x, std::vector<ValueType> const* b, std::vector<ValueType>& result) const override;

            void setSettings(GmmxxLinearEquationSolverSettings<ValueType> const& newSettings);
            GmmxxLinearEquationSolverSettings<ValueType> const& getSettings() const;


            virtual void clearCache() const override;

        private:
            /*!
             * Solves the linear equation system A*x = b given by the parameters using the Jacobi method.
             *
             * @param x The solution vector x. The initial values of x represent a guess of the real values to the
             * solver, but may be set to zero.
             * @param b The right-hand side of the equation system.
             * @return The number of iterations needed until convergence if the solver converged and
             * maximalNumberOfIteration otherwise.
             */
            uint_fast64_t solveLinearEquationSystemWithJacobi(std::vector<ValueType>& x, std::vector<ValueType> const& b) const;
            
            virtual uint64_t getMatrixRowCount() const override;
            virtual uint64_t getMatrixColumnCount() const override;

            // If the solver takes posession of the matrix, we store the moved matrix in this member, so it gets deleted
            // when the solver is destructed.
            std::unique_ptr<storm::storage::SparseMatrix<ValueType>> localA;

            // A pointer to the original sparse matrix given to this solver. If the solver takes posession of the matrix
            // the pointer refers to localA.
            storm::storage::SparseMatrix<ValueType> const* A;
            
            // The settings used by the solver.
            GmmxxLinearEquationSolverSettings<ValueType> settings;
            
            // cached data obtained during solving
            mutable std::unique_ptr<gmm::csr_matrix<ValueType>> gmmxxA;
            mutable std::unique_ptr<gmm::ilu_precond<gmm::csr_matrix<ValueType>>> iluPreconditioner;
            mutable std::unique_ptr<gmm::diagonal_precond<gmm::csr_matrix<ValueType>>> diagonalPreconditioner;
            mutable std::unique_ptr<std::pair<gmm::csr_matrix<ValueType>, std::vector<ValueType>>> jacobiDecomposition;
        };
        
        template<typename ValueType>
        class GmmxxLinearEquationSolverFactory : public LinearEquationSolverFactory<ValueType> {
        public:
            virtual std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const override;
            virtual std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType>&& matrix) const override;
            
            GmmxxLinearEquationSolverSettings<ValueType>& getSettings();
            GmmxxLinearEquationSolverSettings<ValueType> const& getSettings() const;
            
            virtual std::unique_ptr<LinearEquationSolverFactory<ValueType>> clone() const override;

        private:
            GmmxxLinearEquationSolverSettings<ValueType> settings;
        };

    } // namespace solver
} // namespace storm

#endif /* STORM_SOLVER_GMMXXLINEAREQUATIONSOLVER_H_ */
