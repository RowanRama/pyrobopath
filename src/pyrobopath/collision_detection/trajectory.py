from __future__ import annotations
from typing import List, Sequence
import numpy as np
import bisect

from pyrobopath.tools.types import ArrayLike
from pyrobopath.tools.utils import pairwise


class TrajectoryPoint(object):
    """Generic trajectory point described as a one dimensional vector"""

    def __init__(self, data: ArrayLike, time: float):
        # Avoid unnecessary copy if data is already a numpy array
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.time = time

    def __lt__(self, other: TrajectoryPoint):
        return self.time < other.time

    def __eq__(self, other: object):
        if isinstance(other, TrajectoryPoint):
            return (self.data == other.data).all() and self.time == other.time
        raise NotImplemented

    def __repr__(self):
        return f"(Time: {self.time}, Point: {self.data})"

    def interp(self, other: TrajectoryPoint, s: float):
        """Interpolate from the this point to 'other' at s : [0, 1]"""
        data = s * (other.data - self.data) + self.data
        time = s * (other.time - self.time) + self.time
        return TrajectoryPoint(data, time)

    def dist(self, other: TrajectoryPoint):
        """The distance between this point and other"""
        return np.linalg.norm(self.data - other.data)


class Trajectory:
    """
    A trajectory represents a sequence of points in time

    Trajectory points should maintain a stricly increasing sorted order
    """

    def __init__(self, points: List[TrajectoryPoint] | None = None):
        self.points: List[TrajectoryPoint] = []
        if points is not None:
            self.points = points
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> TrajectoryPoint:
        if self.idx == len(self.points):
            raise StopIteration
        result = self.points[self.idx]
        self.idx += 1
        return result

    def __add__(self, other) -> Trajectory:
        new = Trajectory()
        new.points = self.points + other.points
        return new

    def __eq__(self, other: object):
        if isinstance(other, Trajectory):
            return all([tp1 == tp2 for tp1, tp2 in zip(self, other)])
        raise NotImplemented

    def __getitem__(self, key):
        return self.points[key]

    def __repr__(self) -> str:
        out = "Trajectory("
        for p in self.points:
            out += str(p) + " "
        out += ")"
        return out

    def start_time(self):
        if not self.points:
            return 0.0
        return self.points[0].time

    def end_time(self):
        if not self.points:
            return 0.0
        return self.points[-1].time

    def elapsed(self):
        if not self.points:
            return 0.0
        return self.points[-1].time - self.points[0].time

    def offset(self, time):
        for p in self.points:
            p.time += time

    def add_traj_point(self, point):
        self.points.append(point)

    def insert_traj_point(self, index, point):
        self.points.insert(index, point)

    def n_points(self):
        return len(self.points)

    def distance(self):
        length = 0.0
        for s, e in pairwise(self.points):
            length += s.dist(e)
        return length

    def get_point_at_time(self, time) -> TrajectoryPoint | None:
        """
        Interpolate the trajectory at 'time'. This function returns 'None'
        for queries outside of the interval [start_time(), end_time()]
        """
        if not self.points:
            return None

        if time < self.start_time() or time > self.end_time():
            return None

        ans = bisect.bisect_left([p.time for p in self.points], time)
        s = self.points[ans]
        e = self.points[ans - 1]

        if s == e:
            return s
        else:
            return s.interp(e, (time - s.time) / (e.time - s.time))

    def slice(self, start, end) -> Trajectory:
        """
        Returns a new trajectory that has been filtered with trajectory points
        in the closed interval [start, end].
        """
        if not self.points:
            return self

        if start > self.end_time() or end < self.start_time():
            return Trajectory()

        # if the start time is the end time, return a single point
        if start == end:
            new_traj = Trajectory()
            start_point = self.get_point_at_time(start)
            if start_point is not None:
                new_traj.add_traj_point(start_point)
            return new_traj

        # Use bisect to find start and end indices efficiently
        times = [p.time for p in self.points]
        start_idx = bisect.bisect_left(times, start)
        end_idx = bisect.bisect_right(times, end)

        new_traj = Trajectory()

        # Add start point (exact match or interpolated)
        if start >= self.start_time() and start <= self.end_time():
            if start_idx < len(times) and abs(times[start_idx] - start) < 1e-10:
                # Exact match at start
                start_point = self.points[start_idx]
            elif start_idx > 0:
                # Interpolate between points
                s = self.points[start_idx - 1]
                e = self.points[start_idx] if start_idx < len(times) else self.points[-1]
                if abs(e.time - s.time) < 1e-10:
                    start_point = s
                else:
                    interp_s = (start - s.time) / (e.time - s.time)
                    start_point = s.interp(e, interp_s)
            else:
                # start_idx == 0, use first point
                start_point = self.points[0]
            new_traj.add_traj_point(start_point)

        # Add all intermediate points (strictly between start and end)
        for i in range(start_idx, min(end_idx, len(self.points))):
            if self.points[i].time > start and self.points[i].time < end:
                new_traj.points.append(self.points[i])

        # Add end point (exact match or interpolated)
        if end >= self.start_time() and end <= self.end_time():
            # Check if we already added this point as the start point
            if new_traj.points and abs(new_traj.points[-1].time - end) < 1e-10:
                pass  # Already added
            elif end_idx > 0 and end_idx <= len(times) and abs(times[end_idx - 1] - end) < 1e-10:
                # Exact match at end
                end_point = self.points[end_idx - 1]
                new_traj.add_traj_point(end_point)
            elif end_idx > 0:
                # Interpolate
                s = self.points[end_idx - 1] if end_idx > 0 else self.points[0]
                e = self.points[end_idx] if end_idx < len(times) else self.points[-1]
                if abs(e.time - s.time) < 1e-10:
                    end_point = e
                else:
                    interp_s = (end - s.time) / (e.time - s.time)
                    end_point = s.interp(e, interp_s)
                new_traj.add_traj_point(end_point)

        return new_traj

    @staticmethod
    def from_const_vel_path(path: Sequence[ArrayLike], velocity, start_time=0.0):
        traj = Trajectory()
        distance_trav = 0.0
        traj.add_traj_point(TrajectoryPoint(path[0], start_time))
        previous = TrajectoryPoint(path[0], start_time)
        for p in path[1:]:
            current = TrajectoryPoint(p, 0.0)
            distance_trav += current.dist(previous)
            current.time = distance_trav / velocity + start_time
            traj.add_traj_point(current)
            previous = current
        return traj
