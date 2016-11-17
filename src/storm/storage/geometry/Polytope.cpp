#include "src/storm/storage/geometry/Polytope.h"

#include <iostream>

#include "src/storm/adapters/CarlAdapter.h"
#include "src/storm/adapters/HyproAdapter.h"
#include "src/storm/storage/geometry/HyproPolytope.h"
#include "src/storm/utility/macros.h"

#include "src/storm/exceptions/NotImplementedException.h"
#include "src/storm/exceptions/IllegalFunctionCallException.h"

namespace storm {
    namespace storage {
        namespace geometry {
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::create(std::vector<storm::storage::geometry::Halfspace<ValueType>> const& halfspaces) {
                return create(halfspaces, boost::none);
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::create(std::vector<Point> const& points) {
                return create(boost::none, points);
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::createUniversalPolytope() {
                return create(std::vector<Halfspace<ValueType>>(), boost::none);
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::createEmptyPolytope() {
                return create(boost::none, std::vector<Point>());
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::create(boost::optional<std::vector<Halfspace<ValueType>>> const& halfspaces,
                                                                             boost::optional<std::vector<Point>> const& points) {
#ifdef STORM_HAVE_HYPRO
                return HyproPolytope<ValueType>::create(halfspaces, points);
#endif
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "No polytope implementation specified.");
                return nullptr;
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::createDownwardClosure(std::vector<Point> const& points) {
                if(points.empty()) {
                    // In this case, the downwardclosure is empty
                    return createEmptyPolytope();
                }
                uint_fast64_t const dimensions = points.front().size();
                std::vector<Halfspace<ValueType>> halfspaces;
                // We build the convex hull of the given points.
                // However, auxiliary points (that will always be in the downward closure) are added.
                // Then, the halfspaces of the resulting polytope are a superset of the halfspaces of the downward closure.
                std::vector<Point> auxiliaryPoints = points;
                auxiliaryPoints.reserve(auxiliaryPoints.size()*(1+dimensions));
                for(auto const& point : points) {
                    for(uint_fast64_t dim=0; dim<dimensions; ++dim) {
                        auxiliaryPoints.push_back(point);
                        auxiliaryPoints.back()[dim] -= storm::utility::one<ValueType>();
                    }
                }
                std::vector<Halfspace<ValueType>> auxiliaryHalfspaces = create(auxiliaryPoints)->getHalfspaces();
                // The downward closure is obtained by selecting the halfspaces for which the normal vector is non-negative (coordinate wise).
                // Note that due to the auxiliary points, the polytope is never degenerated and thus there is always one unique halfspace-representation which is necessary:
                // Consider, e.g., the convex hull of two points (1,0,0) and (0,1,1).
                // There are multiple halfspace-representations for this set. In particular, there is one where all but one normalVectors have negative entries.
                // However, the downward closure of this set can only be represented with 5 halfspaces.
                for(auto& h : auxiliaryHalfspaces){
                    bool allGreaterZero = true;
                    for(auto const& value : h.normalVector()) {
                        allGreaterZero &= (value >= storm::utility::zero<ValueType>());
                    }
                    if(allGreaterZero){
                        halfspaces.push_back(std::move(h));
                    }
                }
                return create(halfspaces);
            }
            
            template <typename ValueType>
            Polytope<ValueType>::Polytope() {
                // Intentionally left empty
            }
            
            template <typename ValueType>
            Polytope<ValueType>::~Polytope() {
                // Intentionally left empty
            }
            
            template <typename ValueType>
            std::vector<typename Polytope<ValueType>::Point> Polytope<ValueType>::getVertices() const{
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return std::vector<Point>();
            }
            
#ifdef STORM_HAVE_CARL
            template <>
            std::vector<typename Polytope<storm::RationalNumber>::Point> Polytope<storm::RationalNumber>::getVerticesInClockwiseOrder() const{
                std::vector<Point> vertices = getVertices();
                if(vertices.size()<=2) {
                    // In this case, every ordering is clockwise
                    return vertices;
                }
                STORM_LOG_THROW(vertices.front().size()==2, storm::exceptions::IllegalFunctionCallException, "Getting Vertices in clockwise order is only possible for a 2D-polytope.");
                
                std::vector<storm::storage::BitVector> neighborsOfVertices(vertices.size(), storm::storage::BitVector(vertices.size(), false));
                std::vector<Halfspace<storm::RationalNumber>> halfspaces = getHalfspaces();
                for(auto const& h : halfspaces) {
                    storm::storage::BitVector verticesOnHalfspace(vertices.size(), false);
                    for(uint_fast64_t v = 0; v<vertices.size(); ++v) {
                        if(h.isPointOnBoundary(vertices[v])) {
                            verticesOnHalfspace.set(v);
                        }
                    }
                    for(auto const& v : verticesOnHalfspace) {
                        neighborsOfVertices[v] |= verticesOnHalfspace;
                        neighborsOfVertices[v].set(v, false);
                    }
                }
                
                std::vector<Point> result;
                result.reserve(vertices.size());
                storm::storage::BitVector unprocessedVertices(vertices.size(), true);
                
                uint_fast64_t currentVertex = 0;
                for(uint_fast64_t v = 1; v<vertices.size(); ++v) {
                    if(vertices[v].front() < vertices[currentVertex].front()) {
                        currentVertex = v;
                    }
                }
                STORM_LOG_ASSERT(neighborsOfVertices[currentVertex].getNumberOfSetBits()==2, "For 2D Polytopes with at least 3 vertices, each vertex should have exactly 2 neighbors");
                uint_fast64_t firstNeighbor = neighborsOfVertices[currentVertex].getNextSetIndex(0);
                uint_fast64_t secondNeighbor = neighborsOfVertices[currentVertex].getNextSetIndex(firstNeighbor+1);
                uint_fast64_t previousVertex = vertices[firstNeighbor].back() <= vertices[secondNeighbor].back() ? firstNeighbor : secondNeighbor;
                do {
                    unprocessedVertices.set(currentVertex, false);
                    result.push_back(std::move(vertices[currentVertex]));
                    
                    STORM_LOG_ASSERT(neighborsOfVertices[currentVertex].getNumberOfSetBits()==2, "For 2D Polytopes with at least 3 vertices, each vertex should have exactly 2 neighbors");
                    firstNeighbor = neighborsOfVertices[currentVertex].getNextSetIndex(0);
                    secondNeighbor = neighborsOfVertices[currentVertex].getNextSetIndex(firstNeighbor+1);
                    uint_fast64_t nextVertex = firstNeighbor != previousVertex ? firstNeighbor : secondNeighbor;
                    previousVertex = currentVertex;
                    currentVertex = nextVertex;
                } while(!unprocessedVertices.empty());
                
                return result;
            }
#endif
            

            template <typename ValueType>
            std::vector<typename Polytope<ValueType>::Point> Polytope<ValueType>::getVerticesInClockwiseOrder() const{
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                // Note that the implementation for RationalNumber above only works for exact ValueType since
                // checking whether the distance between halfspace and point is zero is problematic otherwise.
                return std::vector<Point>();
            }
            
            template <typename ValueType>
            std::vector<Halfspace<ValueType>> Polytope<ValueType>::getHalfspaces() const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return std::vector<Halfspace<ValueType>>();
            }
            
            template <typename ValueType>
            bool Polytope<ValueType>::isEmpty() const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return false;
            }
            
            template <typename ValueType>
            bool Polytope<ValueType>::isUniversal() const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return false;
            }
            
            template <typename ValueType>
            bool Polytope<ValueType>::contains(Point const& point) const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return false;
            }
            
            template <typename ValueType>
            bool Polytope<ValueType>::contains(std::shared_ptr<Polytope<ValueType>> const& other) const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return false;
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::intersection(std::shared_ptr<Polytope<ValueType>> const& rhs) const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return nullptr;
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::intersection(Halfspace<ValueType> const& halfspace) const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return nullptr;
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::convexUnion(std::shared_ptr<Polytope<ValueType>> const& rhs) const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return nullptr;
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::minkowskiSum(std::shared_ptr<Polytope<ValueType>> const& rhs) const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return nullptr;
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::affineTransformation(std::vector<Point> const& matrix, Point const& vector) const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return nullptr;
            }
            
            template <typename ValueType>
            std::shared_ptr<Polytope<ValueType>> Polytope<ValueType>::downwardClosure() const {
                return createDownwardClosure(this->getVertices());
            }
            
            template <typename ValueType>
            std::pair<typename Polytope<ValueType>::Point, bool> Polytope<ValueType>::optimize(Point const& direction) const {
                STORM_LOG_THROW(false, storm::exceptions::NotImplementedException, "Functionality not implemented.");
                return std::make_pair(Point(), false);
            }
            
            template <typename ValueType>
            template <typename TargetType>
            std::shared_ptr<Polytope<TargetType>> Polytope<ValueType>::convertNumberRepresentation() const {
                if(isEmpty()) {
                    return Polytope<TargetType>::createEmptyPolytope();
                }
                auto halfspaces = getHalfspaces();
                std::vector<Halfspace<TargetType>> halfspacesPrime;
                halfspacesPrime.reserve(halfspaces.size());
                for(auto const& h : halfspaces) {
                    halfspacesPrime.emplace_back(storm::utility::vector::convertNumericVector<TargetType>(h.normalVector()), storm::utility::convertNumber<TargetType>(h.offset()));
                }
                
                return Polytope<TargetType>::create(halfspacesPrime);
            }
            
            template <typename ValueType>
            std::string Polytope<ValueType>::toString(bool numbersAsDouble) const {
                auto halfspaces = this->getHalfspaces();
                std::stringstream stream;
                stream << "Polytope with " << halfspaces.size() << " Halfspaces" << (halfspaces.empty() ? "" : ":") << std::endl;
                for (auto const& h : halfspaces) {
                    stream << "   " << h.toString(numbersAsDouble) << std::endl;
                }
                return stream.str();
            }
            
            template <typename ValueType>
            bool Polytope<ValueType>::isHyproPolytope() const {
                return false;
            }
            
            template class Polytope<double>;
            template std::shared_ptr<Polytope<double>> Polytope<double>::convertNumberRepresentation() const;
            
#ifdef STORM_HAVE_CARL
            template class Polytope<storm::RationalNumber>;
            template std::shared_ptr<Polytope<double>> Polytope<storm::RationalNumber>::convertNumberRepresentation() const;
            template std::shared_ptr<Polytope<storm::RationalNumber>> Polytope<double>::convertNumberRepresentation() const;
            template std::shared_ptr<Polytope<storm::RationalNumber>> Polytope<storm::RationalNumber>::convertNumberRepresentation() const;
#endif
        }
    }
}
