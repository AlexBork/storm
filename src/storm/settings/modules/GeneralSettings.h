#ifndef STORM_SETTINGS_MODULES_GENERALSETTINGS_H_
#define STORM_SETTINGS_MODULES_GENERALSETTINGS_H_

#include "storm-config.h"
#include "storm/settings/modules/ModuleSettings.h"

#include "storm/builder/ExplorationOrder.h"

namespace storm {
    namespace settings {
        namespace modules {

            /*!
             * This class represents the general settings.
             */
            class GeneralSettings : public ModuleSettings {
            public:

                /*!
                 * Creates a new set of general settings.
                 */
                GeneralSettings();

                /*!
                 * Retrieves whether the help option was set.
                 *
                 * @return True if the help option was set.
                 */
                bool isHelpSet() const;

                /*!
                 * Retrieves whether the version option was set.
                 *
                 * @return True if the version option was set.
                 */
                bool isVersionSet() const;

                /*!
                 * Retrieves the name of the module for which to show the help or "all" to indicate that the full help
                 * needs to be shown.
                 *
                 * @return The name of the module for which to show the help or "all".
                 */
                std::string getHelpModuleName() const;

                /*!
                 * Retrieves whether the verbose option was set.
                 *
                 * @return True if the verbose option was set.
                 */
                bool isVerboseSet() const;

                /*!
                 * Retrieves the precision to use for numerical operations.
                 *
                 * @return The precision to use for numerical operations.
                 */
                double getPrecision() const;

                /*!
                 * Retrieves whether the config option was set.
                 *
                 * @return True if the config option was set.
                 */
                bool isConfigSet() const;

                /*!
                 * Retrieves the name of the file that is to be scanned for settings.
                 *
                 * @return The name of the file that is to be scanned for settings.
                 */
                std::string getConfigFilename() const;

                /*!
                 * Retrieves whether the property option was set.
                 *
                 * @return True if the property option was set.
                 */
                bool isPropertySet() const;

                /*!
                 * Retrieves the property specified with the property option.
                 *
                 * @return The property specified with the property option.
                 */
                std::string getProperty() const;

                /*!
                 * Retrieves whether the option to perform bisimulation minimization is set.
                 *
                 * @return True iff the option was set.
                 */
                bool isBisimulationSet() const;

                /*!
                 * Retrieves whether the option enabling parametric model checking is set.
                 *
                 * @return True iff the option was set.
                 */
                bool isParametricSet() const;

                
                /*!
                 * Retrieves whether the option enabling parametric region model checking is set.
                 *
                 * @return True iff the option was set.
                 */
                bool isParametricRegionSet() const;

                /*!
                 * Retrieves whether a min/max equation solving technique has been set.
                 *
                 * @return True iff an equation solving technique has been set.
                 */
                bool isMinMaxEquationSolvingTechniqueSet() const;
                
                /*!
                 * Retrieves whether time and memory consumption shall be printed at the end of a run.
                 *
                 * @return True iff the option was set.
                 */
                bool isPrintTimeAndMemorySet() const;

                /*!
                 * Retrieves whether the option enabling exact model checking is set.
                 *
                 * @return True iff the option was set.
                 */
                bool isExactSet() const;

                bool check() const override;
                void finalize() override;

                // The name of the module.
                static const std::string moduleName;

            private:
                // Define the string names of the options as constants.
                static const std::string helpOptionName;
                static const std::string helpOptionShortName;
                static const std::string printTimeAndMemoryOptionName;
                static const std::string printTimeAndMemoryOptionShortName;
                static const std::string versionOptionName;
                static const std::string verboseOptionName;
                static const std::string verboseOptionShortName;
                static const std::string precisionOptionName;
                static const std::string precisionOptionShortName;
                static const std::string configOptionName;
                static const std::string configOptionShortName;
                static const std::string propertyOptionName;
                static const std::string propertyOptionShortName;
                static const std::string bisimulationOptionName;
                static const std::string bisimulationOptionShortName;
                static const std::string parametricOptionName;

                static const std::string parametricRegionOptionName;

                static const std::string exactOptionName;
            };

        } // namespace modules
    } // namespace settings
} // namespace storm

#endif /* STORM_SETTINGS_MODULES_GENERALSETTINGS_H_ */
