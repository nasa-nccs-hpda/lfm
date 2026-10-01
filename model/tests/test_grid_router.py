import math
import unittest

from lfm.model.grid_registry import GridFamily, default_grid_registry
from lfm.model.grid_router import (
    GeographicRoutingError,
    normalize_lunar_longitude,
    route_aoi,
    route_point,
)


class GridRouterTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = default_grid_registry()

    def test_point_routes_at_both_polar_thresholds(self):
        cases = (
            (81.999999, 10, "24N", GridFamily.LTM),
            (82.0, 10, "LPS_N", GridFamily.LPS_N),
            (-81.999999, 10, "24S", GridFamily.LTM),
            (-82.0, 10, "LPS_S", GridFamily.LPS_S),
        )
        for lat, lon, grid_id, family in cases:
            with self.subTest(lat=lat):
                route = route_point(lat=lat, lon=lon, registry=self.registry)
                self.assertEqual(route.grid_id, grid_id)
                self.assertIs(route.family, family)

    def test_point_at_poles_uses_canonical_zero_longitude(self):
        north = route_point(lat=90, lon=137, registry=self.registry)
        south = route_point(lat=-90, lon=-44, registry=self.registry)

        self.assertEqual((north.grid_id, north.lon), ("LPS_N", 0.0))
        self.assertEqual((south.grid_id, south.lon), ("LPS_S", 0.0))

    def test_ltm_point_edges_and_equator_are_deterministic(self):
        cases = (
            (1, -180, "1N"),
            (1, -172, "2N"),
            (1, 180, "45N"),
            (-1, -180, "1S"),
            (0, 149.7, "42N"),
        )
        for lat, lon, grid_id in cases:
            with self.subTest(lat=lat, lon=lon):
                self.assertEqual(
                    route_point(
                        lat=lat,
                        lon=lon,
                        registry=self.registry,
                    ).grid_id,
                    grid_id,
                )

    def test_every_ltm_longitude_boundary_has_one_owner(self):
        for boundary_number in range(46):
            longitude = -180 + boundary_number * 8
            expected_zone = min(45, boundary_number + 1)
            with self.subTest(longitude=longitude):
                route = route_point(
                    lat=1,
                    lon=longitude,
                    registry=self.registry,
                )
                self.assertEqual(route.grid_id, f"{expected_zone}N")

    def test_longitude_normalization_is_finite_and_endpoint_aware(self):
        self.assertEqual(normalize_lunar_longitude(190), -170)
        self.assertEqual(normalize_lunar_longitude(-190), 170)
        self.assertEqual(normalize_lunar_longitude(540), 180)
        self.assertEqual(normalize_lunar_longitude(-540), -180)
        with self.assertRaises(GeographicRoutingError):
            normalize_lunar_longitude(math.inf)

    def test_aoi_crossing_north_threshold_is_partitioned(self):
        parts = route_aoi(
            ul_lat=83,
            ul_lon=149,
            lr_lat=81,
            lr_lon=151,
            registry=self.registry,
        )

        self.assertEqual(tuple(part.grid_id for part in parts), ("42N", "LPS_N"))
        self.assertEqual(
            tuple((part.south, part.north) for part in parts),
            ((81.0, 82.0), (82.0, 83.0)),
        )

    def test_aoi_touching_threshold_does_not_add_boundary_only_grid(self):
        north_touch = route_aoi(
            ul_lat=82,
            ul_lon=149,
            lr_lat=81,
            lr_lon=151,
            registry=self.registry,
        )
        polar_only = route_aoi(
            ul_lat=83,
            ul_lon=149,
            lr_lat=82,
            lr_lon=151,
            registry=self.registry,
        )

        self.assertEqual(tuple(part.grid_id for part in north_touch), ("42N",))
        self.assertEqual(tuple(part.grid_id for part in polar_only), ("LPS_N",))

    def test_aoi_crossing_south_threshold_is_partitioned(self):
        parts = route_aoi(
            ul_lat=-81,
            ul_lon=149,
            lr_lat=-83,
            lr_lon=151,
            registry=self.registry,
        )

        self.assertEqual(tuple(part.grid_id for part in parts), ("42S", "LPS_S"))
        self.assertEqual(
            tuple((part.south, part.north) for part in parts),
            ((-82.0, -81.0), (-83.0, -82.0)),
        )

    def test_aoi_touching_south_threshold_has_one_family(self):
        south_touch = route_aoi(
            ul_lat=-81,
            ul_lon=149,
            lr_lat=-82,
            lr_lon=151,
            registry=self.registry,
        )
        polar_only = route_aoi(
            ul_lat=-82,
            ul_lon=149,
            lr_lat=-83,
            lr_lon=151,
            registry=self.registry,
        )

        self.assertEqual(tuple(part.grid_id for part in south_touch), ("42S",))
        self.assertEqual(tuple(part.grid_id for part in polar_only), ("LPS_S",))

    def test_aoi_crossing_equator_splits_ltm_hemispheres(self):
        parts = route_aoi(
            ul_lat=1,
            ul_lon=149,
            lr_lat=-1,
            lr_lon=151,
            registry=self.registry,
        )

        self.assertEqual(tuple(part.grid_id for part in parts), ("42N", "42S"))
        self.assertEqual(
            tuple((part.south, part.north) for part in parts),
            ((0.0, 1.0), (-1.0, 0.0)),
        )

    def test_nonpolar_antimeridian_aoi_uses_edge_ltm_zones(self):
        parts = route_aoi(
            ul_lat=2,
            ul_lon=178,
            lr_lat=1,
            lr_lon=-178,
            registry=self.registry,
        )

        self.assertEqual(tuple(part.grid_id for part in parts), ("1N", "45N"))
        self.assertEqual(
            tuple((part.west, part.east) for part in parts),
            ((-180.0, -178.0), (178.0, 180.0)),
        )

    def test_polar_antimeridian_aoi_keeps_two_query_parts(self):
        parts = route_aoi(
            ul_lat=85,
            ul_lon=178,
            lr_lat=84,
            lr_lon=-178,
            registry=self.registry,
        )

        self.assertEqual(tuple(part.grid_id for part in parts), ("LPS_N", "LPS_N"))
        self.assertEqual(
            tuple((part.west, part.east) for part in parts),
            ((-180.0, -178.0), (178.0, 180.0)),
        )
        self.assertEqual(len(parts), len(set(parts)))

    def test_full_longitude_is_limited_to_one_polar_cap(self):
        cases = (
            (90, 82, "LPS_N"),
            (-82, -90, "LPS_S"),
        )
        for north, south, grid_id in cases:
            with self.subTest(grid_id=grid_id):
                parts = route_aoi(
                    ul_lat=north,
                    ul_lon=-180,
                    lr_lat=south,
                    lr_lon=180,
                    registry=self.registry,
                )
                self.assertEqual(len(parts), 1)
                self.assertEqual(
                    (parts[0].grid_id, parts[0].west, parts[0].east),
                    (grid_id, -180.0, 180.0),
                )

        with self.assertRaisesRegex(GeographicRoutingError, "polar cap"):
            route_aoi(
                ul_lat=10,
                ul_lon=-180,
                lr_lat=9,
                lr_lon=180,
                registry=self.registry,
            )
        with self.assertRaisesRegex(GeographicRoutingError, "polar cap"):
            route_aoi(
                ul_lat=85,
                ul_lon=-180,
                lr_lat=83,
                lr_lon=180,
                registry=self.registry,
            )

    def test_ambiguous_and_world_spanning_longitudes_are_rejected(self):
        for west, east in ((0, 180), (-170, 170), (90, -90)):
            with self.subTest(west=west, east=east):
                with self.assertRaises(GeographicRoutingError):
                    route_aoi(
                        ul_lat=2,
                        ul_lon=west,
                        lr_lat=1,
                        lr_lon=east,
                        registry=self.registry,
                    )

    def test_exact_ltm_longitude_edges_have_no_duplicate_parts(self):
        parts = route_aoi(
            ul_lat=2,
            ul_lon=-172,
            lr_lat=1,
            lr_lon=-164,
            registry=self.registry,
        )

        self.assertEqual(tuple(part.grid_id for part in parts), ("2N",))
        self.assertEqual(len(parts), len(set(parts)))

    def test_routing_order_is_stable(self):
        kwargs = {
            "ul_lat": 83,
            "ul_lon": 178,
            "lr_lat": 81,
            "lr_lon": -178,
            "registry": self.registry,
        }
        self.assertEqual(route_aoi(**kwargs), route_aoi(**kwargs))

    def test_invalid_geographic_inputs_are_rejected(self):
        invalid_points = (
            (91, 0),
            (-91, 0),
            (0, math.nan),
            (math.inf, 0),
            (0, 10**400),
        )
        for lat, lon in invalid_points:
            with self.subTest(lat=lat, lon=lon):
                with self.assertRaises(GeographicRoutingError):
                    route_point(lat=lat, lon=lon, registry=self.registry)

        with self.assertRaisesRegex(GeographicRoutingError, "ul_lat > lr_lat"):
            route_aoi(
                ul_lat=1,
                ul_lon=10,
                lr_lat=1,
                lr_lon=11,
                registry=self.registry,
            )


if __name__ == "__main__":
    unittest.main()
