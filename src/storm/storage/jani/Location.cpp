#include "storm/storage/jani/Location.h"

#include "storm/utility/macros.h"
#include "storm/exceptions/InvalidJaniException.h"
#include "storm/exceptions/InvalidArgumentException.h"

namespace storm {
    namespace jani {
        
        Location::Location(std::string const& name, std::vector<Assignment> const& transientAssignments) : name(name), assignments(transientAssignments) {
            // Intentionally left empty.
        }
        
        std::string const& Location::getName() const {
            return name;
        }
        
        OrderedAssignments const& Location::getAssignments() const {
            return assignments;
        }
        
        void Location::addTransientAssignment(storm::jani::Assignment const& assignment) {
            STORM_LOG_THROW(assignment.isTransient(), storm::exceptions::InvalidArgumentException, "Must not add non-transient assignment to location.");
            assignments.add(assignment);
        }
        
        void Location::substitute(std::map<storm::expressions::Variable, storm::expressions::Expression> const& substitution) {
            for (auto& assignment : assignments) {
                assignment.substitute(substitution);
            }
        }
        
        void Location::checkValid() const {
            // Intentionally left empty.
        }
        
    }
}
