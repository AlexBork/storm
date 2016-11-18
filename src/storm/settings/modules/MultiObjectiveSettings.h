#ifndef STORM_SETTINGS_MODULES_MULTIOBJECTIVESETTINGS_H_
#define STORM_SETTINGS_MODULES_MULTIOBJECTIVESETTINGS_H_

#include "storm/settings/modules/ModuleSettings.h"

namespace storm {
    namespace settings {
        namespace modules {

            /*!
             * This class represents the settings for multi-objective model checking.
             */
            class MultiObjectiveSettings : public ModuleSettings {
            public:
                
                /*!
                 * Creates a new set of multi-objective model checking settings.
                 */
                MultiObjectiveSettings();
				
				/*!
				 * Retrieves whether the data for plotting should be exported.
                 * @return  True iff the data for plotting should be exported.
                 */
				bool isExportPlotSet() const;
				
				/*!
				 * The path to a directory in which the plot data should be stored.
                 * @return A path to a directory.
                 */
				std::string getExportPlotDirectory() const;
				
				/**
				 * Retrieves the desired precision for quantitative- and pareto queries.
                 * @return the desired precision for quantitative- and pareto queries.
				 */
				double getPrecision() const;
				
                /*!
                 * Retrieves whether or not a threshold for the number of performed refinement steps is given.
                 *
                 * @return True if a threshold for the number of performed refinement steps is given.
                 */
                bool isMaxStepsSet() const;
                
                
                /*!
                 * Retrieves The maximum number of refinement steps that should be performed (if given).
                 *
                 * @return the maximum number of refinement steps that should be performed (if given).
                 */
                uint_fast64_t getMaxSteps() const;
                
                
                /*!
                 * Checks whether the settings are consistent. If they are inconsistent, an exception is thrown.
                 *
                 * @return True if the settings are consistent.
                 */
                virtual bool check() const override;
                
                
                const static std::string moduleName;
                
            private:
				const static std::string exportPlotOptionName;
				const static std::string precisionOptionName;
				const static std::string maxStepsOptionName;
            };
            
        } // namespace modules
    } // namespace settings
} // namespace storm

#endif /* STORM_SETTINGS_MODULES_MULTIOBJECTIVESETTINGS_H_ */
