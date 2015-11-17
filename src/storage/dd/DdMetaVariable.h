#ifndef STORM_STORAGE_DD_DDMETAVARIBLE_H_
#define STORM_STORAGE_DD_DDMETAVARIBLE_H_

#include <vector>

#include "src/storage/dd/DdType.h"
#include "src/storage/dd/Bdd.h"
#include "src/storage/dd/AddIterator.h"

namespace storm {
    namespace dd {
        template<DdType LibraryType>
        class DdManager;
        
        template<DdType LibraryType, typename ValueTpe>
        class Add;

        // An enumeration for all legal types of meta variables.
        enum class MetaVariableType { Bool, Int };
        
        // Declare DdMetaVariable class so we can then specialize it for the different DD types.
        template<DdType LibraryType>
        class DdMetaVariable {
        public:
            friend class DdManager<LibraryType>;
            friend class Bdd<LibraryType>;

            template <DdType LibraryTypePrime, typename ValueTypePrime>
            friend class Add;

            template <DdType LibraryTypePrime, typename ValueTypePrime>
            friend class AddIterator;
            
            /*!
             * Retrieves the name of the meta variable.
             *
             * @return The name of the variable.
             */
            std::string const& getName() const;
            
            /*!
             * Retrieves the type of the meta variable.
             *
             * @return The type of the meta variable.
             */
            MetaVariableType getType() const;
            
            /*!
             * Retrieves the lowest value of the range of the variable.
             *
             * @return The lowest value of the range of the variable.
             */
            int_fast64_t getLow() const;
            
            /*!
             * Retrieves the highest value of the range of the variable.
             *
             * @return The highest value of the range of the variable.
             */
            int_fast64_t getHigh() const;

            /*!
             * Retrieves the number of DD variables for this meta variable.
             *
             * @return The number of DD variables for this meta variable.
             */
            std::size_t getNumberOfDdVariables() const;

            /*!
             * Retrieves the cube of all variables that encode this meta variable.
             *
             * @return The cube of all variables that encode this meta variable.
             */
            Bdd<LibraryType> const& getCube() const;
            
        private:
            /*!
             * Creates an integer meta variable with the given name and range bounds.
             *
             * @param name The name of the meta variable.
             * @param low The lowest value of the range of the variable.
             * @param high The highest value of the range of the variable.
             * @param ddVariables The vector of variables used to encode this variable.
             * @param manager A pointer to the manager that is responsible for this meta variable.
             */
            DdMetaVariable(std::string const& name, int_fast64_t low, int_fast64_t high, std::vector<Bdd<LibraryType>> const& ddVariables);
            
            /*!
             * Creates a boolean meta variable with the given name.
             * @param name The name of the meta variable.
             * @param ddVariables The vector of variables used to encode this variable.
             * @param manager A pointer to the manager that is responsible for this meta variable.
             */
            DdMetaVariable(std::string const& name, std::vector<Bdd<LibraryType>> const& ddVariables);
            
            /*!
             * Retrieves the variables used to encode the meta variable.
             *
             * @return A vector of variables used to encode the meta variable.
             */
            std::vector<Bdd<LibraryType>> const& getDdVariables() const;
            
            /*!
             * Creates the cube for this meta variable from the DD variables.
             */
            void createCube();
            
            // The name of the meta variable.
            std::string name;
            
            // The type of the variable.
            MetaVariableType type;
            
            // The lowest value of the range of the variable.
            int_fast64_t low;
            
            // The highest value of the range of the variable.
            int_fast64_t high;
            
            // The vector of variables that are used to encode the meta variable.
            std::vector<Bdd<DdType::CUDD>> ddVariables;
            
            // The cube consisting of all variables that encode the meta variable.
            Bdd<DdType::CUDD> cube;
        };
    }
}

#endif /* STORM_STORAGE_DD_DDMETAVARIBLE_H_ */