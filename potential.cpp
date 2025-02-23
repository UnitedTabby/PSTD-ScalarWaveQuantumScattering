/* 
 * MIT License
 * 
 *  
 * Copyright (c) 2025 Kun Chen <kunchen@siom.ac.cn>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 */

#include <cmath>
#include "run_environment.h"

// A spherically symmetric potential barrier,
// the height of the potential is set to 5% 
// the neutron energy (which is given in the input example),
// and the radius of the potential is set to the 
// neutron wavelength ($=2\pi\lambdaba$).
#define V0	0.327216805 * 0.05
#define RMAX2	6.283185307179586476925286766559005768394 * 6.283185307179586476925286766559005768394

QPrecision V(QPrecision x, QPrecision y, QPrecision z)
{
   return (((x*x+y*y+z*z)<=RMAX2)?V0:0.0);
}
