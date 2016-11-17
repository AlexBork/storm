#include "Place.h"

#include "src/storm/exceptions/IllegalArgumentValueException.h"
#include "src/storm/utility/macros.h"

namespace storm {
    namespace gspn {
        Place::Place(uint64_t id) : id(id) {

        }

        void Place::setName(std::string const& name) {
            this->name = name;
        }

        std::string Place::getName() const {
            return this->name;
        }

        uint64_t Place::getID() const {
            return this->id;
        }

        void Place::setNumberOfInitialTokens(uint64_t tokens) {
            this->numberOfInitialTokens = tokens;
        }

        uint64_t Place::getNumberOfInitialTokens() const {
            return this->numberOfInitialTokens;
        }

        void Place::setCapacity(uint64_t cap) {
            this->capacity = cap;
        }

        uint64_t Place::getCapacity() const {
            return capacity.get();
        }
        
        bool Place::hasRestrictedCapacity() const {
            return capacity != boost::none;
        }
    }
}
