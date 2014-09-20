#ifndef STORM_SETTINGS_ARGUMENTBASE_H_
#define STORM_SETTINGS_ARGUMENTBASE_H_

#include <iostream>
#include <string>
#include <boost/algorithm/string.hpp>

#include "src/settings/ArgumentType.h"

namespace storm {
	namespace settings {

        /*!
         * This class serves as the (untemplated) base class of argument classes.
         */
		class ArgumentBase {
		public:
            /*!
             * Constructs a new argument base with the given name, description and indication whether the argument is
             * optional.
             *
             * @param name The name of the argument.
             * @param description A description of the argument.
             * @param isOptional A flag indicating whether the argument is optional.
             */
			ArgumentBase(std::string const& name, std::string const& description) : hasBeenSet(false), name(name), description(description) {
                // Intentionally left empty.
            }
            
            /*!
             * Retrieves the type of the argument.
             *
             * @return The type of the argument.
             */
			virtual ArgumentType getType() const = 0;

            /*!
             * Retrieves whether the argument is optional.
             *
             * @return True iff the argument is optional.
             */
			virtual bool getIsOptional() const = 0;

            /*!
             * Retrieves the name of the argument.
             *
             * @return The name of the argument.
             */
			std::string const& getName() const {
				return this->name;
			}

            /*!
             * Retrieves the description of the argument.
             *
             * @return The description of the argument.
             */
			std::string const& getDescription() const {
				return this->description;
			}

            /*!
             * Retrieves whether the argument has a default value.
             *
             * @return True iff the argument has a default value.
             */
			virtual bool getHasDefaultValue() const = 0;
            
            /*!
             * Retrieves whether the argument has been set.
             *
             * @return True iff the argument has been set.
             */
			virtual bool getHasBeenSet() const {
				return this->hasBeenSet;
			}

            /*!
             * Sets the value of the argument from the default value.
             */
			virtual void setFromDefaultValue() = 0;

            /*!
             * Tries to set the value of the argument from the given string.
             *
             * @param stringValue The new value of the argument given as a string.
             * @return True iff the assignment was successful.
             */
			virtual bool setFromStringValue(std::string const& stringValue) = 0;

            /*!
             * Retrieves the value of this argument as a string.
             *
             * @return The value of this argument as a string.
             */
			virtual std::string getValueAsString() const = 0;
            
            /*!
             * Retrieves the value of this argument as an integer. If the conversion cannot be performed, an exception
             * is thrown.
             *
             * @return The value of this argument as an integer.
             */
			virtual int_fast64_t getValueAsInteger() const = 0;
            
            /*!
             * Retrieves the value of this argument as an unsigned integer. If the conversion cannot be performed, an
             * exception is thrown.
             *
             * @return The value of this argument as an unsigned integer.
             */
			virtual uint_fast64_t getValueAsUnsignedInteger() const = 0;

            /*!
             * Retrieves the value of this argument as a double. If the conversion cannot be performed, an exception
             * is thrown.
             *
             * @return The value of this argument as an double.
             */
			virtual double getValueAsDouble() const = 0;
            
            /*!
             * Retrieves the value of this argument as a boolean. If the conversion cannot be performed, an exception
             * is thrown.
             *
             * @return The value of this argument as an boolean.
             */
			virtual bool getValueAsBoolean() const = 0;
            
		protected:            
            // A flag indicating whether the argument has been set.
			bool hasBeenSet;

            // The name of the argument.
			std::string name;
            
            // The description of the argument.
			std::string description;

            /*!
             * Converts the given value represented as a string to the type of the template parameter. The second
             * is used to signal that the conversion was successful (or not).
             *
             * @param valueAsString The value to convert.
             * @param conversionSuccessful After a call to this function returned, the supplied boolean indicates
             * whether the conversion was successful.
             */
            template <typename TargetType>
            static TargetType convertFromString(std::string const& valueAsString, bool& conversionSuccessful);
            
            /*!
             * Converts the given value to a string representation.
             *
             * @param value The value to convert.
             * @return The string representation of the value.
             */
            template <typename ValueType>
            static std::string convertToString(ValueType const& value);
		};

        template <typename TargetType>
        TargetType ArgumentBase::convertFromString(std::string const& valueAsString, bool& conversionSuccessful) {
            std::istringstream stream(valueAsString);
            TargetType t;
            conversionSuccessful = (stream >> t) && (stream >> std::ws).eof();
            return t;
        }
        
        template <>
        bool ArgumentBase::convertFromString<bool>(std::string const& s, bool& ok) {
            static const std::string lowerTrueString = "true";
            static const std::string lowerFalseString = "false";
            static const std::string lowerYesString = "yes";
            static const std::string lowerNoString = "no";
            
            std::string lowerInput = boost::algorithm::to_lower_copy(s);
            
            if (s.compare(lowerTrueString) == 0 || s.compare(lowerYesString) == 0) {
                ok = true;
                return true;
            } else if (s.compare(lowerFalseString) == 0 || s.compare(lowerNoString) == 0) {
                ok = true;
                return false;
            }
            
            std::istringstream stream(s);
            bool t;
            ok = (stream >> t) && (stream >> std::ws).eof();
            return t;
        }
        
        template <typename ValueType>
        std::string ArgumentBase::convertToString(ValueType const& value) {
            std::ostringstream stream;
            stream << value;
            return stream.str();
        }
        
	}
}

#endif // STORM_SETTINGS_ARGUMENTBASE_H_