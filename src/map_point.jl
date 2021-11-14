mutable struct MapPoint
    id::Int64   # id of 3d mappoint
    kfid::Int64   # Id of the KeyFrame, from which it was created.
    point_type::Int64  # person landmark, 物体sift

    # Set of KeyFrame's ids that are visible from this MapPoint.
    observer_keyframes_ids::OrderedSet{Int64}
    descriptor::BitVector   # Mean descriptor.
    keyframes_descriptors::Dict{Int64, BitVector}
    # Descriptor distances to each KeyFrame. kfid => hamming_distance.
    descriptor_distances_map::Dict{Int64, Float64}
    position::Point3f   # Position in world coordinate system.
    is_3d::Bool   # True if the MapPoint has been initialized.
    # True if the MapPoint is visible in the current Frame.
    is_observed::Bool
end

function MapPoint(id, kfid, descriptor, is_observed::Bool = true)
    observed_keyframes_ids = OrderedSet{Int64}(kfid)
    keyframes_descriptors = Dict{Int64, BitVector}(kfid => descriptor)
    descriptor_distances_map = Dict{Int64, Float64}(kfid => 0.0)
    position = Point3f(0, 0, 0)
    is_3d = false

    MapPoint(
        id, kfid, observed_keyframes_ids,
        descriptor, keyframes_descriptors, descriptor_distances_map,
        position, is_3d, is_observed)
end

@inline is_valid(m::MapPoint)::Bool = m.id != -1

function add_keyframe_observation!(m::MapPoint, kfid)
    push!(m.observer_keyframes_ids, kfid)
end

function get_observers(m::MapPoint)
    deepcopy(m.observer_keyframes_ids)
end

function get_observers_number(m::MapPoint)
    length(m.observer_keyframes_ids)
end

function get_position(m::MapPoint)
    m.position
end

function set_position!(m::MapPoint, position)
    m.position = position
    m.is_3d = true
end

function remove_kf_observation!(m::MapPoint, kfid)
    kfid in keys(m.observer_keyframes_ids) || return
    pop!(m.observer_keyframes_ids, kfid)
    if isempty(m.observer_keyframes_ids)
        m.descriptor |> empty!
        m.keyframes_descriptors |> empty!
        m.descriptor_distances_map |> empty!
        return
    end
    # Set new anchor KeyFrame if removed previous anchor.
    kfid == m.kfid && (m.kfid = m.observer_keyframes_ids[1];)
    # Find the most representative descriptor among observer KeyFrames.
    mindist = length(m.descriptor) * 8.0
    minid = -1

    kfid in keys(m.keyframes_descriptors) || return
    kfid_desc = m.keyframes_descriptors[kfid]
    for (kfd, kfd_desc) in m.keyframes_descriptors
        kfd == kfid && continue
        dist = hamming_distance(kfid_desc, kfd_desc)
        m.descriptor_distances_map[kfd] -= dist

        desc_distance = m.descriptor_distances_map[kfd]
        if desc_distance < mindist
            mindist = desc_distance
            minid = kfd
        end
    end

    pop!(m.keyframes_descriptors, kfid)
    pop!(m.descriptor_distances_map, kfid)
    minid > 0 && (m.descriptor = m.keyframes_descriptors[minid])
end

function add_descriptor!(m::MapPoint, kfid, descriptor::BitVector)
    kfid in keys(m.keyframes_descriptors) && return

    descriptor_distance = 0.0
    m.keyframes_descriptors[kfid] = descriptor
    m.descriptor_distances_map[kfid] = descriptor_distance
    length(m.keyframes_descriptors) == 1 && (m.descriptor = descriptor; return)
    # Find the most representative descriptor among observer KeyFrames.
    mindist = length(m.descriptor) * 8.0
    minid = -1
    for (kfd, kfd_desc) in m.keyframes_descriptors
        dist = hamming_distance(descriptor, kfd_desc)
        m.descriptor_distances_map[kfd] += dist
        dist < mindist && (mindist = dist; minid = kfd;)
        descriptor_distance += dist
    end
    # Get descriptor with minimal distance.
    descriptor_distance < mindist && (minid = kfid;)
    m.descriptor = m.keyframes_descriptors[minid]
end

function mappoint_min_distance(m1::MapPoint, m2::MapPoint)
    min_distance = 1e6
    for d1 in values(m1.keyframes_descriptors), d2 in values(m2.keyframes_descriptors)
        distance = hamming_distance(d1, d2)
        distance < min_distance && (min_distance = distance;)
    end

    min_distance
end
