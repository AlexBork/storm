#pragma once

#include "storm-config.h"
#include "src/settings/modules/ModuleSettings.h"

#include "src/solver/SolverSelectionOptions.h"

namespace storm {
    namespace settings {
        namespace modules {
            
            /*!
             * This class represents the min/max solver settings.
             */
            class MinMaxEquationSolverSettings : public ModuleSettings {
            public:
                // An enumeration of all available convergence criteria.
                enum class ConvergenceCriterion { Absolute, Relative };
                
                MinMaxEquationSolverSettings();
              
                /*!
                 * Retrieves whether a min/max equation solving technique has been set.
                 *
                 * @return True iff an equation solving technique has been set.
                 */
                bool isMinMaxEquationSolvingMethodSet() const;
                
                /*!
                 * Retrieves the selected min/max equation solving technique.
                 *
                 * @return The selected min/max equation solving technique.
                 */
                storm::solver::MinMaxMethod getMinMaxEquationSolvingMethod() const;
                
                /*!
                 * Retrieves whether the maximal iteration count has been set.
                 *
                 * @return True iff the maximal iteration count has been set.
                 */
                bool isMaximalIterationCountSet() const;
                
                /*!
                 * Retrieves the maximal number of iterations to perform until giving up on converging.
                 *
                 * @return The maximal iteration count.
                 */
                uint_fast64_t getMaximalIterationCount() const;
                
                /*!
                 * Retrieves whether the precision has been set.
                 *
                 * @return True iff the precision has been set.
                 */
                bool isPrecisionSet() const;
                
                /*!
                 * Retrieves the precision that is used for detecting convergence.
                 *
                 * @return The precision to use for detecting convergence.
                 */
                double getPrecision() const;
                
                /*!
                 * Retrieves whether the convergence criterion has been set.
                 *
                 * @return True iff the convergence criterion has been set.
                 */
                bool isConvergenceCriterionSet() const;
                
                /*!
                 * Retrieves the selected convergence criterion.
                 *
                 * @return The selected convergence criterion.
                 */
                ConvergenceCriterion getConvergenceCriterion() const;
                
                // The name of the module.
                static const std::string moduleName;
                
            private:
                static const std::string solvingMethodOptionName;
                static const std::string maximalIterationsOptionName;
                static const std::string maximalIterationsOptionShortName;
                static const std::string precisionOptionName;
                static const std::string absoluteOptionName;
            };
            
        }
    }
}