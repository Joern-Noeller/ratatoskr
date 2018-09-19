////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2018 Jan Moritz Joseph
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////
#include "Packet.h"

int Packet::idcnt=0;

Packet::Packet(Node* src, Node* dst, int size, int generationTime, int trafficTypeId, int as, int ac) {
	this->id = idcnt;
	idcnt++;

	this->src = src;
	this->dst = dst;
	this->size = size;
	this->generationTime = generationTime;
	this->dbid = rep.registerElement("Packet", id);
	this->pkgclass = -1;
	this->numhops  = 0;
	this->as = as;
	this->ac= ac;
	this->trafficTypeId = trafficTypeId;

	rep.reportAttribute(dbid, "packet_src", std::to_string(src->id));
	rep.reportAttribute(dbid, "packet_dst", std::to_string(dst->id));
}

Packet::~Packet() {
	for (Flit* f : toTransmit) {
		delete f;
	}
	for (Flit* f : inTransmit) {
		delete f;
	}
	for (Flit* f : transmitted) {
		delete f;
	}
}

std::ostream& operator << (std::ostream& os, const Packet& p) {
		os << "ID: " << p.id << ", SRC: " << p.src->id << ", DST: " << p.dst->id << ", Size: " << p.size << ", Generated at: "<<p.generationTime << std::endl;
		return os;
	}