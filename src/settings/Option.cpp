#include "src/settings/Option.h"

#include <iomanip>
#include <string>
#include <algorithm>
#include "ArgumentBase.h"
#include "Argument.h"

#include "src/utility/macros.h"
#include "src/exceptions/IllegalArgumentException.h"
#include "src/exceptions/OptionUnificationException.h"


namespace storm {
    namespace settings {
       
			Option::Option(std::string const& moduleName, std::string const& longOptionName, std::string const& optionDescription, bool isOptionRequired, bool requireModulePrefix, std::vector<std::shared_ptr<ArgumentBase>> const& optionArguments) : Option(moduleName, longOptionName, "", false, optionDescription, isOptionRequired, requireModulePrefix, optionArguments) {
				// Intentionally left empty.
			}
            
            Option::Option(std::string const& moduleName, std::string const& longOptionName, std::string const& shortOptionName, std::string const& optionDescription, bool isOptionRequired, bool requireModulePrefix, std::vector<std::shared_ptr<ArgumentBase>> const& optionArguments) : Option(moduleName, longOptionName, shortOptionName, true, optionDescription, isOptionRequired, requireModulePrefix, optionArguments) {
                // Intentionally left empty.
            }

         
            bool Option::isCompatibleWith(Option const& other) {
                STORM_LOG_THROW(this->getArgumentCount() == other.getArgumentCount(), storm::exceptions::OptionUnificationException, "Unable to unify two options, because their argument count differs.");

				for(size_t i = 0; i != this->arguments.size(); i++) {
                    ArgumentBase const& firstArgument = this->getArgument(i);
                    ArgumentBase const& secondArgument = other.getArgument(i);

                    STORM_LOG_THROW(firstArgument.getType() == secondArgument.getType(), storm::exceptions::OptionUnificationException, "Unable to unify two options, because their arguments are incompatible.");

					switch (firstArgument.getType()) {
						case ArgumentType::String:
							static_cast<storm::settings::Argument<std::string> const&>(firstArgument).isCompatibleWith(static_cast<storm::settings::Argument<std::string> const&>(secondArgument));
							break;
						case ArgumentType::Integer:
							static_cast<storm::settings::Argument<int_fast64_t> const&>(firstArgument).isCompatibleWith(static_cast<storm::settings::Argument<int_fast64_t> const&>(secondArgument));
							break;
						case ArgumentType::UnsignedInteger:
							static_cast<storm::settings::Argument<uint_fast64_t> const&>(firstArgument).isCompatibleWith(static_cast<storm::settings::Argument<uint_fast64_t> const&>(secondArgument));
							break;
						case ArgumentType::Double:
							static_cast<storm::settings::Argument<double> const&>(firstArgument).isCompatibleWith(static_cast<storm::settings::Argument<double> const&>(secondArgument));
							break;
						case ArgumentType::Boolean:
							static_cast<storm::settings::Argument<bool> const&>(firstArgument).isCompatibleWith(static_cast<storm::settings::Argument<bool> const&>(secondArgument));
							break;
					}
				}
                return true;
			}

           
			uint_fast64_t Option::getArgumentCount() const {
				return this->arguments.size();
			}

           
			ArgumentBase const& Option::getArgument(uint_fast64_t argumentIndex) const {
                STORM_LOG_THROW(argumentIndex < this->getArgumentCount(), storm::exceptions::IllegalArgumentException, "Index of argument is out of bounds.");
				return *this->arguments.at(argumentIndex);
			}
            
			ArgumentBase& Option::getArgument(uint_fast64_t argumentIndex) {
                STORM_LOG_THROW(argumentIndex < this->getArgumentCount(), storm::exceptions::IllegalArgumentException, "Index of argument is out of bounds.");
				return *this->arguments.at(argumentIndex);
			}

			
			ArgumentBase const& Option::getArgumentByName(std::string const& argumentName) const {
				auto argumentIterator = this->argumentNameMap.find(argumentName);
                STORM_LOG_THROW(argumentIterator != this->argumentNameMap.end(), storm::exceptions::IllegalArgumentException, "Unable to retrieve argument with unknown name '" << argumentName << "'.");
				return *argumentIterator->second;
			}

			std::string const& Option::getLongName() const {
				return this->longName;
			}

       
            bool Option::getHasShortName() const {
                return this->hasShortName;
            }
            
           
			std::string const& Option::getShortName() const {
				return this->shortName;
			}

			std::string const& Option::getDescription() const {
				return this->description;
			}

			std::string const& Option::getModuleName() const {
				return this->moduleName;
			}

			bool Option::getIsRequired() const {
				return this->isRequired;
			}

           
            bool Option::getRequiresModulePrefix() const {
                return this->requireModulePrefix;
            }
            
			bool Option::getHasOptionBeenSet() const {
				return this->hasBeenSet;
			}

            Option::Option(std::string const& moduleName, std::string const& longOptionName, std::string const& shortOptionName, bool hasShortOptionName, std::string const& optionDescription, bool isOptionRequired, bool requireModulePrefix, std::vector<std::shared_ptr<ArgumentBase>> const& optionArguments) : longName(longOptionName), hasShortName(hasShortOptionName), shortName(shortOptionName), description(optionDescription), moduleName(moduleName), isRequired(isOptionRequired), requireModulePrefix(requireModulePrefix), hasBeenSet(false), arguments(optionArguments), argumentNameMap() {
                // First, do some sanity checks.
                STORM_LOG_THROW(!longName.empty(), storm::exceptions::IllegalArgumentException, "Unable to construct option with empty name.");
                STORM_LOG_THROW(!moduleName.empty(), storm::exceptions::IllegalArgumentException, "Unable to construct option with empty module name.");
                
                bool longNameContainsNonAlpha = std::find_if(longName.begin(), longName.end(), [](char c) { return !(std::isalpha(c) || std::isdigit(c) || c == '-'); }) != longName.end();
                STORM_LOG_THROW(!longNameContainsNonAlpha, storm::exceptions::IllegalArgumentException, "Unable to construct option with illegal long name '" << longName << "'.");
                
                bool shortNameContainsNonAlpha = std::find_if(shortName.begin(), shortName.end(), [](char c) { return !(std::isalpha(c) || std::isdigit(c) || c == '-'); }) != shortName.end();
                STORM_LOG_THROW(!shortNameContainsNonAlpha, storm::exceptions::IllegalArgumentException, "Unable to construct option with illegal short name '" << shortName << "'.");

                // Then index all arguments.
                for (auto const& argument : arguments) {
                    argumentNameMap.emplace(argument->getName(), argument);
                }
            }
            
            void  Option::setHasOptionBeenSet(bool newValue) {
                this->hasBeenSet = newValue;
            }
        
            uint_fast64_t Option::getPrintLength() const {
                uint_fast64_t length = 2;
                if (!this->getRequiresModulePrefix()) {
                    length += 2;
                }
                length += this->getModuleName().length() + 1;
                length += this->getLongName().length();
                if (this->getHasShortName()) {
                    length += 4;
                    if (!this->getRequiresModulePrefix()) {
                        length += 2;
                    }
                    length += this->getModuleName().length() + 1;
                    length += this->getShortName().length();
                }
                return length;
            }
        
        std::vector<std::shared_ptr<ArgumentBase>> const& Option::getArguments() const {
            return this->arguments;
        }
        
        std::ostream& operator<<(std::ostream& out, Option const& option) {
            uint_fast64_t width = static_cast<uint_fast64_t>(out.width());
            
            uint_fast64_t charactersPrinted = 0;
            out << std::setw(0) << "--";
            charactersPrinted += 2;
            if (!option.getRequiresModulePrefix()) {
                out << "[";
                ++charactersPrinted;
            }
            out <<  option.getModuleName() << ":";
            charactersPrinted += option.getModuleName().length() + 1;
            if (!option.getRequiresModulePrefix()) {
                out << "]";
                ++charactersPrinted;
            }
            out << option.getLongName();
            charactersPrinted += option.getLongName().length();
            if (option.getHasShortName()) {
                out << " | -";
                charactersPrinted += 4;
                if (!option.getRequiresModulePrefix()) {
                    out << "[";
                    ++charactersPrinted;
                }
                out << option.getModuleName() << ":";
                charactersPrinted += option.getModuleName().length() + 1;
                if (!option.getRequiresModulePrefix()) {
                    out << "]";
                    ++charactersPrinted;
                }
                out << option.getShortName();
                charactersPrinted += option.getShortName().length();
            }
            
            // Now fill the width.
            for (uint_fast64_t i = charactersPrinted; i < width; ++i) {
                out << out.fill();
            }
            
            out << "\t" << option.getDescription();
            
            if (option.getArgumentCount() > 0) {
                // Start by determining the longest print length of the arguments.
                uint_fast64_t maxLength = 0;
                for (auto const& argument : option.getArguments()) {
                    maxLength = std::max(maxLength, argument->getPrintLength());
                }
                
                for (auto const& argument : option.getArguments()) {
                    out << std::endl;
                    out << "\t* " << std::setw(maxLength) << std::left << *argument;
                }
            }
            
            return out;
        }
    }
}
