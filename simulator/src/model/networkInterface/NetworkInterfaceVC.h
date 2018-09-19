/*******************************************************************************
 * Copyright (C) 2018 joseph
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/
#pragma once

#include "systemc.h"
#include <queue>
#include <algorithm>

#include "utils/Structures.h"
#include "model/traffic/Flit.h"
#include <model/container/FlitContainer.h>
#include <model/container/PacketContainer.h>
#include <utils/GlobalReportClass.h>

#include "NetworkInterface.h"

class NetworkInterfaceVC : public NetworkInterface{
public:
	std::queue < Packet* > packet_send_queue;
	std::queue < Flit* > flit_queue;
	std::queue < Packet* > packet_recv_queue;


	std::vector<bool>* flowControlOut;
	std::vector<int>* tagOut;
	std::vector<bool>* emptyOut;
	sc_in < bool > clk;
	FlitPortContainer* flitPortContainer;
	PacketPortContainer* packetPortContainer;

	GlobalReportClass& report = GlobalReportClass::getInstance();


	SC_HAS_PROCESS(NetworkInterfaceVC);
	NetworkInterfaceVC(sc_module_name nm, Node* node);
	~NetworkInterfaceVC();

	void initialize();
	void bind(Connection*, SignalContainer*, SignalContainer*);
	void thread();
	void receivePacket();
	void receiveFlit();

};
