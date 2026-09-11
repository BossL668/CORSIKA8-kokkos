/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace corsika::terrain {

  inline MeshData MeshLoader::loadOBJ(std::string const& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
      throw std::runtime_error("MeshLoader::loadOBJ: Cannot open file: " + filepath);
    }

    MeshData data;
    std::string line;

    while (std::getline(file, line)) {
      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') { continue; }

      std::istringstream iss(line);
      std::string prefix;
      iss >> prefix;

      if (prefix == "v") {
        // Vertex line: v x y z
        double x, y, z;
        if (!(iss >> x >> y >> z)) {
          throw std::runtime_error("MeshLoader::loadOBJ: Invalid vertex line: " + line);
        }
        data.vertices.push_back({x, y, z});
      } else if (prefix == "f") {
        // Face line: f v1 v2 v3 [v4 ...]
        // Indices in OBJ are 1-based, may include texture/normal indices: v/vt/vn
        std::vector<size_t> faceIndices;
        std::string token;

        while (iss >> token) {
          // Parse vertex index (before first slash if present)
          size_t slashPos = token.find('/');
          std::string indexStr =
              (slashPos != std::string::npos) ? token.substr(0, slashPos) : token;

          int idx = std::stoi(indexStr);
          // Convert from 1-based to 0-based, handle negative indices
          if (idx > 0) {
            faceIndices.push_back(static_cast<size_t>(idx - 1));
          } else if (idx < 0) {
            // Negative indices are relative to current vertex count
            faceIndices.push_back(data.vertices.size() + idx);
          } else {
            throw std::runtime_error("MeshLoader::loadOBJ: Invalid index 0 in face: " +
                                     line);
          }
        }

        // Triangulate polygon using fan triangulation
        if (faceIndices.size() < 3) {
          throw std::runtime_error(
              "MeshLoader::loadOBJ: Face with less than 3 vertices: " + line);
        }

        for (size_t i = 1; i + 1 < faceIndices.size(); ++i) {
          data.faces.push_back({faceIndices[0], faceIndices[i], faceIndices[i + 1]});
        }
      }
      // Ignore other lines (vt, vn, mtllib, usemtl, etc.)
    }

    if (data.vertices.empty()) {
      throw std::runtime_error("MeshLoader::loadOBJ: No vertices found in file: " +
                               filepath);
    }

    if (data.faces.empty()) {
      throw std::runtime_error("MeshLoader::loadOBJ: No faces found in file: " +
                               filepath);
    }

    return data;
  }

  inline MeshData MeshLoader::loadPLY(std::string const& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("MeshLoader::loadPLY: Cannot open file: " + filepath);
    }

    // Verify PLY magic
    std::string line;
    std::getline(file, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "ply") {
      throw std::runtime_error("MeshLoader::loadPLY: Not a PLY file: " + filepath);
    }

    // Parse header
    bool isBinaryLE = false;
    size_t vertexCount = 0;
    size_t faceCount = 0;

    struct PropInfo {
      std::string name;
      std::string type;
    };
    std::vector<PropInfo> vertexProps;
    std::string faceCountType, faceIndexType;

    enum class Elem { None, Vertex, Face, Other };
    Elem current = Elem::None;

    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line == "end_header") break;

      std::istringstream iss(line);
      std::string token;
      iss >> token;

      if (token == "format") {
        std::string fmt;
        iss >> fmt;
        if (fmt == "binary_little_endian") {
          isBinaryLE = true;
        } else {
          throw std::runtime_error(
              "MeshLoader::loadPLY: Only binary_little_endian supported, got: " + fmt);
        }
      } else if (token == "element") {
        std::string name;
        size_t count;
        iss >> name >> count;
        if (name == "vertex") {
          current = Elem::Vertex;
          vertexCount = count;
        } else if (name == "face") {
          current = Elem::Face;
          faceCount = count;
        } else {
          current = Elem::Other;
        }
      } else if (token == "property") {
        std::string typeOrList;
        iss >> typeOrList;
        if (typeOrList == "list") {
          std::string countType, indexType, propName;
          iss >> countType >> indexType >> propName;
          if (current == Elem::Face) {
            faceCountType = countType;
            faceIndexType = indexType;
          }
        } else {
          std::string propName;
          iss >> propName;
          if (current == Elem::Vertex) { vertexProps.push_back({propName, typeOrList}); }
        }
      }
    }

    if (!isBinaryLE) {
      throw std::runtime_error(
          "MeshLoader::loadPLY: Only binary_little_endian format supported");
    }

    auto plyTypeSize = [](std::string const& t) -> size_t {
      if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
      if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
      if (t == "int" || t == "uint" || t == "int32" || t == "uint32") return 4;
      if (t == "float" || t == "float32") return 4;
      if (t == "double" || t == "float64" || t == "int64" || t == "uint64") return 8;
      throw std::runtime_error("MeshLoader::loadPLY: Unknown PLY type: " + t);
      return 0;
    };

    // Determine vertex byte layout; find x, y, z offsets
    size_t vertexByteSize = 0;
    size_t xOffset = 0, yOffset = 0, zOffset = 0;
    std::string xType, yType, zType;
    bool foundX = false, foundY = false, foundZ = false;

    for (auto const& prop : vertexProps) {
      size_t sz = plyTypeSize(prop.type);
      if (prop.name == "x") {
        xOffset = vertexByteSize;
        xType = prop.type;
        foundX = true;
      } else if (prop.name == "y") {
        yOffset = vertexByteSize;
        yType = prop.type;
        foundY = true;
      } else if (prop.name == "z") {
        zOffset = vertexByteSize;
        zType = prop.type;
        foundZ = true;
      }
      vertexByteSize += sz;
    }

    if (!foundX || !foundY || !foundZ) {
      throw std::runtime_error(
          "MeshLoader::loadPLY: Missing x, y, or z vertex properties in: " + filepath);
    }

    auto readScalar = [](std::vector<char> const& buf, size_t offset,
                         std::string const& type) -> double {
      if (type == "float" || type == "float32") {
        float v;
        std::memcpy(&v, buf.data() + offset, sizeof(float));
        return static_cast<double>(v);
      }
      if (type == "double" || type == "float64") {
        double v;
        std::memcpy(&v, buf.data() + offset, sizeof(double));
        return v;
      }
      throw std::runtime_error(
          "MeshLoader::loadPLY: Expected float/double for coordinate, got: " + type);
      return 0.0;
    };

    MeshData data;
    data.vertices.reserve(vertexCount);

    std::vector<char> vertBuf(vertexByteSize);
    for (size_t i = 0; i < vertexCount; ++i) {
      file.read(vertBuf.data(), static_cast<std::streamsize>(vertexByteSize));
      if (!file) {
        throw std::runtime_error(
            "MeshLoader::loadPLY: Unexpected EOF reading vertices in: " + filepath);
      }
      data.vertices.push_back({readScalar(vertBuf, xOffset, xType),
                               readScalar(vertBuf, yOffset, yType),
                               readScalar(vertBuf, zOffset, zType)});
    }

    data.faces.reserve(faceCount);
    size_t const countTypeSize = plyTypeSize(faceCountType);
    size_t const indexTypeSize = plyTypeSize(faceIndexType);

    for (size_t i = 0; i < faceCount; ++i) {
      // Read per-face vertex count
      uint8_t count = 0;
      if (countTypeSize == 1) {
        file.read(reinterpret_cast<char*>(&count), 1);
      } else {
        std::vector<char> tmp(countTypeSize, 0);
        file.read(tmp.data(), static_cast<std::streamsize>(countTypeSize));
        count = static_cast<uint8_t>(static_cast<unsigned char>(tmp[0]));
      }
      if (!file) {
        throw std::runtime_error(
            "MeshLoader::loadPLY: Unexpected EOF reading face count in: " + filepath);
      }
      if (count < 3) {
        throw std::runtime_error(
            "MeshLoader::loadPLY: Face with fewer than 3 vertices in: " + filepath);
      }

      std::vector<size_t> indices(count);
      for (uint8_t j = 0; j < count; ++j) {
        if (indexTypeSize == 4) {
          uint32_t idx = 0;
          file.read(reinterpret_cast<char*>(&idx), 4);
          indices[j] = static_cast<size_t>(idx);
        } else if (indexTypeSize == 8) {
          uint64_t idx = 0;
          file.read(reinterpret_cast<char*>(&idx), 8);
          indices[j] = static_cast<size_t>(idx);
        } else {
          throw std::runtime_error(
              "MeshLoader::loadPLY: Unsupported face index type size in: " + filepath);
        }
      }
      if (!file) {
        throw std::runtime_error(
            "MeshLoader::loadPLY: Unexpected EOF reading face indices in: " + filepath);
      }

      // Fan triangulate
      for (uint8_t j = 1; j + 1 < count; ++j) {
        data.faces.push_back({indices[0], indices[j], indices[j + 1]});
      }
    }

    if (data.vertices.empty()) {
      throw std::runtime_error("MeshLoader::loadPLY: No vertices found in file: " +
                               filepath);
    }
    if (data.faces.empty()) {
      throw std::runtime_error("MeshLoader::loadPLY: No faces found in file: " +
                               filepath);
    }

    return data;
  }

  inline MeshData MeshLoader::load(std::string const& filepath) {
    // Detect format from extension
    std::string ext;
    size_t dotPos = filepath.rfind('.');
    if (dotPos != std::string::npos) {
      ext = filepath.substr(dotPos);
      // Convert to lowercase
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return std::tolower(c); });
    }

    if (ext == ".obj") {
      return loadOBJ(filepath);
    } else if (ext == ".ply") {
      return loadPLY(filepath);
    } else {
      throw std::runtime_error("MeshLoader::load: Unsupported file format: " + ext);
    }
  }

  inline std::vector<Point> MeshLoader::toPoints(MeshData const& data,
                                                 CoordinateSystemPtr cs,
                                                 LengthType scale) {
    std::vector<Point> points;
    points.reserve(data.vertices.size());

    for (auto const& v : data.vertices) {
      points.emplace_back(cs, v[0] * scale, v[1] * scale, v[2] * scale);
    }

    return points;
  }

} // namespace corsika

